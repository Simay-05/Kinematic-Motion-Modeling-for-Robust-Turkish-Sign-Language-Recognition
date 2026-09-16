from __future__ import annotations
import argparse, copy, itertools, json, random, time
from pathlib import Path
from typing import Dict, Optional
import cv2, joblib, numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
EPS = 1e-08
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}
KEEP = list(range(17)) + list(range(91, 112)) + list(range(112, 133))
N_KEYPOINTS = 59
RUNS = ['mmpose_rf', 'mmpose_svm', 'mmpose_mlp', 'mmpose_lstm']

def safe_stem(value):
    text = str(value).strip().replace('\\', '/').split('/')[-1]
    for ext in list(VIDEO_EXTS) + ['.npy']:
        if text.lower().endswith(ext):
            return text[:-len(ext)]
    return text

def find_videos(data_dir: Path) -> Dict[str, Path]:
    lookup = {}
    for path in data_dir.rglob('*'):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            lookup.setdefault(safe_stem(path.name), path)
    print(f'[videos] Found {len(lookup)} videos.')
    return lookup

def match_video(sample_id, lookup):
    sample = safe_stem(sample_id)
    for candidate in [sample, sample + '_color', sample.replace('_rgb', '_color'), sample.replace('_color', ''), sample.replace('_depth', '_color')]:
        if candidate in lookup:
            return lookup[candidate]
    return None

def load_manifest(path: Path):
    df = pd.read_csv(path)
    required = {'sample_id', 'label', 'split'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Manifest missing: {sorted(missing)}')
    df['sample_id'] = df['sample_id'].astype(str)
    df['label'] = df['label'].astype(str)
    df['split'] = df['split'].astype(str).str.lower()
    df.loc[df['split'].isin(['valid', 'validation']), 'split'] = 'val'
    return df.reset_index(drop=True)

def choose_person(instances):
    if not instances:
        return None

    def score(instance):
        if 'bbox_score' in instance:
            try:
                return float(np.asarray(instance['bbox_score']).reshape(-1)[0])
            except Exception:
                pass
        if 'keypoint_scores' in instance:
            try:
                return float(np.mean(instance['keypoint_scores']))
            except Exception:
                pass
        return 0.0
    return max(instances, key=score)

def parse_result(result, width, height):
    out = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
    batches = result.get('predictions', [])
    if not batches:
        return out
    person = choose_person(batches[0])
    if person is None:
        return out
    keypoints = np.asarray(person.get('keypoints', []), dtype=np.float32)
    scores = np.asarray(person.get('keypoint_scores', []), dtype=np.float32).reshape(-1)
    if keypoints.ndim == 3:
        keypoints = keypoints[0]
    if len(keypoints) < 133:
        return out
    selected = keypoints[KEEP, :2]
    out[:, 0] = selected[:, 0] / max(width, 1)
    out[:, 1] = selected[:, 1] / max(height, 1)
    out[:, 2] = scores[KEEP] if len(scores) >= 133 else 1.0
    out[~np.isfinite(out)] = 0.0
    return out

def extract_video(inferencer, path, stride, max_frames):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open {path}')
    seq, frame_i, used = ([], 0, 0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_i % stride != 0:
            frame_i += 1
            continue
        h, w = frame.shape[:2]
        result = next(inferencer(frame, show=False, return_vis=False))
        seq.append(parse_result(result, w, h))
        used += 1
        frame_i += 1
        if used >= max_frames:
            break
    cap.release()
    return np.stack(seq) if seq else np.zeros((1, N_KEYPOINTS, 3), dtype=np.float32)

def extract_all(args, manifest):
    try:
        from mmpose.apis import MMPoseInferencer
    except ImportError as exc:
        raise ImportError('Install MMPose first.') from exc
    lookup = find_videos(args.data_dir)
    args.landmarks_dir.mkdir(parents=True, exist_ok=True)
    print('[mmpose] Loading official wholebody alias...')
    inferencer = MMPoseInferencer(pose2d='wholebody', device=args.mmpose_device)
    failures = []
    for i, row in manifest.iterrows():
        sample_id = row['sample_id']
        out = args.landmarks_dir / f'{safe_stem(sample_id)}.npy'
        if out.exists() and (not args.overwrite):
            continue
        video = match_video(sample_id, lookup)
        if video is None:
            failures.append((sample_id, 'video_missing'))
            continue
        if i % 10 == 0:
            print(f'[extract] {i}/{len(manifest)} {sample_id}')
        try:
            np.save(out, extract_video(inferencer, video, args.frame_stride, args.max_frames))
        except Exception as exc:
            failures.append((sample_id, str(exc)))
    if failures:
        pd.DataFrame(failures, columns=['sample_id', 'error']).to_csv(args.landmarks_dir / 'failures.csv', index=False)
    print(f"[extract] Saved {len(list(args.landmarks_dir.glob('*.npy')))} landmark files.")
    print(f'[extract] Failures: {len(failures)}')

def detection_rate(seq):
    return float((seq[:, :, 2].max(axis=1) > 0.05).mean())

def quality_filter(manifest, landmarks_dir, threshold):
    keep, rejected = ([], [])
    for i, row in manifest.iterrows():
        path = landmarks_dir / f"{safe_stem(row['sample_id'])}.npy"
        if not path.exists():
            rejected.append((row['sample_id'], 'missing'))
            continue
        try:
            rate = detection_rate(np.load(path, mmap_mode='r'))
        except Exception as exc:
            rejected.append((row['sample_id'], f'load_error:{exc}'))
            continue
        if rate < threshold:
            rejected.append((row['sample_id'], f'detection_rate={rate:.3f}'))
        else:
            keep.append(i)
    return (manifest.loc[keep].reset_index(drop=True), rejected)

def normalize(seq):
    xy = np.asarray(seq, dtype=np.float32)[:, :, :2].copy()
    xy[~np.isfinite(xy)] = 0.0
    out = np.zeros_like(xy)
    for t, frame in enumerate(xy):
        valid = np.linalg.norm(frame, axis=1) > EPS
        anchors = [i for i in [5, 6, 11, 12] if valid[i]]
        if len(anchors) >= 2:
            center = frame[anchors].mean(axis=0)
        elif valid.any():
            center = frame[valid].mean(axis=0)
        else:
            out[t] = out[t - 1] if t else 0.0
            continue
        centered = frame - center
        scale = np.linalg.norm(frame[5] - frame[6]) if valid[5] and valid[6] else np.median(np.linalg.norm(centered[valid], axis=1))
        if not np.isfinite(scale) or scale < EPS:
            scale = 1.0
        out[t] = centered / scale
        out[t, ~valid] = 0.0
    return out

def resample(seq, frames):
    if len(seq) == frames:
        return seq.astype(np.float32)
    if len(seq) == 1:
        return np.repeat(seq, frames, axis=0).astype(np.float32)
    old, new = (np.linspace(0, 1, len(seq)), np.linspace(0, 1, frames))
    flat = seq.reshape(len(seq), -1)
    result = np.empty((frames, flat.shape[1]), dtype=np.float32)
    for j in range(flat.shape[1]):
        result[:, j] = np.interp(new, old, flat[:, j])
    return result.reshape((frames,) + seq.shape[1:])

def raw_sequence(seq, frames):
    return resample(normalize(seq), frames).reshape(frames, -1).astype(np.float32)

def tabular_vector(seq, segments, sampled_frames):
    pieces = [seq.mean(0), seq.std(0), seq.min(0), seq.max(0), np.median(seq, axis=0)]
    for idx in np.array_split(np.arange(len(seq)), segments):
        pieces.extend([seq[idx].mean(0), seq[idx].std(0)])
    ids = np.linspace(0, len(seq) - 1, sampled_frames).round().astype(int)
    pieces.append(seq[ids].reshape(-1))
    out = np.concatenate(pieces).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out

def build_features(manifest, landmarks_dir, frames, segments, sampled_frames, cache_dir):
    tab_cache = cache_dir / 'tabular.joblib'
    seq_cache = cache_dir / 'sequence.npz'
    if tab_cache.exists() and seq_cache.exists():
        tab = joblib.load(tab_cache)
        seq = np.load(seq_cache, allow_pickle=False)
        return (tab['X'], seq['X'], tab['y'])
    cache_dir.mkdir(parents=True, exist_ok=True)
    Xtab, Xseq, y = ([], [], [])
    for i, row in manifest.iterrows():
        if i % 100 == 0:
            print(f'[features] {i}/{len(manifest)}')
        arr = np.load(landmarks_dir / f"{safe_stem(row['sample_id'])}.npy").astype(np.float32)
        seq = raw_sequence(arr, frames)
        Xseq.append(seq)
        Xtab.append(tabular_vector(seq, segments, sampled_frames))
        y.append(str(row['label']))
    Xtab, Xseq, y = (np.vstack(Xtab), np.stack(Xseq), np.asarray(y))
    joblib.dump({'X': Xtab, 'y': y}, tab_cache)
    np.savez_compressed(seq_cache, X=Xseq, y=y)
    print(f'[features] tabular={Xtab.shape}, sequence={Xseq.shape}')
    return (Xtab, Xseq, y)

def splits(manifest):
    s = manifest['split'].to_numpy()
    return (np.where(s == 'train')[0], np.where(s == 'val')[0], np.where(s == 'test')[0])

def candidates(name, seed):
    if name == 'rf':
        for d, leaf, mf in itertools.product([None, 20, 40], [1, 2, 4], ['sqrt', 0.3]):
            p = dict(n_estimators=600, max_depth=d, min_samples_leaf=leaf, max_features=mf, class_weight='balanced_subsample', n_jobs=-1, random_state=seed)
            yield (p, RandomForestClassifier(**p))
    elif name == 'svm':
        for var, c, gamma in itertools.product([0.9, 0.95], [1.0, 10.0, 30.0], ['scale', 0.001]):
            p = dict(pca_variance=var, C=c, gamma=gamma)
            yield (p, Pipeline([('scale', StandardScaler()), ('pca', PCA(n_components=var, svd_solver='full')), ('model', SVC(C=c, gamma=gamma, kernel='rbf', class_weight='balanced', probability=True, random_state=seed))]))
    elif name == 'mlp':
        for var, layers, alpha, lr in itertools.product([0.9, 0.95], [(256,), (256, 128)], [0.0001, 0.001], [0.0005, 0.001]):
            p = dict(pca_variance=var, hidden_layers=layers, alpha=alpha, learning_rate=lr)
            yield (p, Pipeline([('scale', StandardScaler()), ('pca', PCA(n_components=var, svd_solver='full')), ('model', MLPClassifier(hidden_layer_sizes=layers, alpha=alpha, batch_size=32, learning_rate_init=lr, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, max_iter=500, random_state=seed))]))

def rebuild(name, p, seed):
    if name == 'rf':
        return RandomForestClassifier(**p)
    if name == 'svm':
        return Pipeline([('scale', StandardScaler()), ('pca', PCA(n_components=p['pca_variance'], svd_solver='full')), ('model', SVC(C=p['C'], gamma=p['gamma'], kernel='rbf', class_weight='balanced', probability=True, random_state=seed))])
    return Pipeline([('scale', StandardScaler()), ('pca', PCA(n_components=p['pca_variance'], svd_solver='full')), ('model', MLPClassifier(hidden_layer_sizes=tuple(p['hidden_layers']), alpha=p['alpha'], batch_size=32, learning_rate_init=p['learning_rate'], early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, max_iter=500, random_state=seed))])

def tune_classical(name, Xtr, ytr, Xv, yv, seed, outdir):
    best_p, best_f1, best_acc = (None, -1.0, -1.0)
    records = []
    models = list(candidates(name, seed))
    print(f'[tuning:{name}] {len(models)} configurations')
    for i, (p, model) in enumerate(models, 1):
        model.fit(Xtr, ytr)
        pred = model.predict(Xv)
        f1 = f1_score(yv, pred, average='macro', zero_division=0)
        acc = accuracy_score(yv, pred)
        records.append({'configuration': i, 'validation_macro_f1': f1, 'validation_accuracy': acc, 'parameters': json.dumps(p)})
        print(f'[tuning:{name}] {i:02d}/{len(models)} f1={f1:.4f} acc={acc:.4f}')
        if f1 > best_f1 or (np.isclose(f1, best_f1) and acc > best_acc):
            best_p, best_f1, best_acc = (p, float(f1), float(acc))
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).sort_values(['validation_macro_f1', 'validation_accuracy'], ascending=False).to_csv(outdir / 'validation_search.csv', index=False)
    return (best_p, best_f1, best_acc)

class AttentionBiLSTM(nn.Module):

    def __init__(self, input_size, classes):
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.proj = nn.Sequential(nn.Linear(input_size, 256), nn.GELU(), nn.Dropout(0.35))
        self.lstm = nn.LSTM(256, 128, 2, batch_first=True, bidirectional=True, dropout=0.35)
        self.attn = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Sequential(nn.LayerNorm(256), nn.Dropout(0.35), nn.Linear(256, classes))

    def forward(self, x):
        x = self.proj(self.norm(x))
        seq, _ = self.lstm(x)
        weights = torch.softmax(self.attn(seq).squeeze(-1), dim=1)
        pooled = torch.sum(seq * weights.unsqueeze(-1), dim=1)
        return self.head(pooled)

def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def save_metrics(ytrue, pred, outdir, extras):
    outdir.mkdir(parents=True, exist_ok=True)
    m = dict(accuracy=float(accuracy_score(ytrue, pred)), macro_f1=float(f1_score(ytrue, pred, average='macro', zero_division=0)), weighted_f1=float(f1_score(ytrue, pred, average='weighted', zero_division=0)), **extras)
    (outdir / 'metrics.json').write_text(json.dumps(m, indent=2))
    (outdir / 'classification_report.txt').write_text(classification_report(ytrue, pred, zero_division=0))
    labels = sorted(set(ytrue) | set(pred), key=str)
    pd.DataFrame(confusion_matrix(ytrue, pred, labels=labels), index=labels, columns=labels).to_csv(outdir / 'confusion_matrix.csv')
    return m

def train_classical(name, manifest, X, y, outdir, seed):
    tr, val, te = splits(manifest)
    p, vf1, vacc = tune_classical(name, X[tr], y[tr], X[val], y[val], seed, outdir)
    fit = np.concatenate([tr, val])
    model = rebuild(name, p, seed)
    t0 = time.perf_counter()
    model.fit(X[fit], y[fit])
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(X[te])
    infer_s = time.perf_counter() - t0
    joblib.dump(model, outdir / 'model.joblib')
    return save_metrics(y[te], pred, outdir, {'extractor': 'mmpose_wholebody_body_hands', 'model': name, 'validation_macro_f1': vf1, 'validation_accuracy': vacc, 'n_features': int(X.shape[1]), 'training_seconds': train_s, 'inference_seconds': infer_s})

def train_lstm(manifest, X, y, outdir, seed, epochs, batch_size):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tr, val, te = splits(manifest)
    enc = LabelEncoder().fit(y[tr])
    yy = enc.transform(y)
    scaler = StandardScaler().fit(X[tr].reshape(-1, X.shape[-1]))

    def scale(idx):
        return scaler.transform(X[idx].reshape(-1, X.shape[-1])).reshape(X[idx].shape).astype(np.float32)
    loaders = {}
    for name, idx, shuffle in [('tr', tr, True), ('val', val, False), ('te', te, False)]:
        ds = TensorDataset(torch.tensor(scale(idx)), torch.tensor(yy[idx], dtype=torch.long))
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    dev = device()
    model = AttentionBiLSTM(X.shape[-1], len(enc.classes_)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0007, weight_decay=0.0002)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, min_lr=1e-05)
    best_state, best_f1, best_loss, stale = (None, -1.0, float('inf'), 0)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in loaders['tr']:
            bx, by = (bx.to(dev), by.to(dev))
            opt.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        total, vp, vt = (0.0, [], [])
        with torch.no_grad():
            for bx, by in loaders['val']:
                bx, by = (bx.to(dev), by.to(dev))
                logits = model(bx)
                total += loss_fn(logits, by).item() * len(bx)
                vp.extend(logits.argmax(1).cpu().numpy())
                vt.extend(by.cpu().numpy())
        vloss = total / max(len(vt), 1)
        vf1 = f1_score(vt, vp, average='macro', zero_division=0)
        scheduler.step(vloss)
        history.append({'epoch': epoch, 'validation_loss': vloss, 'validation_macro_f1': vf1})
        print(f'[mmpose+lstm] epoch={epoch:03d} val_loss={vloss:.4f} val_f1={vf1:.4f}')
        if vf1 > best_f1 + 0.0001 or (np.isclose(vf1, best_f1) and vloss < best_loss):
            best_state, best_f1, best_loss, stale = (copy.deepcopy(model.state_dict()), float(vf1), float(vloss), 0)
        else:
            stale += 1
            if stale >= 12:
                break
    model.load_state_dict(best_state)
    model.eval()
    pred = []
    with torch.no_grad():
        for bx, _ in loaders['te']:
            pred.extend(model(bx.to(dev)).argmax(1).cpu().numpy())
    pred = enc.inverse_transform(np.asarray(pred))
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), outdir / 'attention_bilstm.pt')
    joblib.dump(scaler, outdir / 'scaler.joblib')
    joblib.dump(enc, outdir / 'label_encoder.joblib')
    (outdir / 'history.json').write_text(json.dumps(history, indent=2))
    return save_metrics(y[te], pred, outdir, {'extractor': 'mmpose_wholebody_body_hands', 'model': 'attention_bilstm', 'validation_macro_f1': best_f1, 'best_validation_loss': best_loss, 'device': str(dev), 'sequence_frames': int(X.shape[1]), 'features_per_frame': int(X.shape[2])})

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--landmarks_dir', type=Path, default=Path('mmpose_landmarks'))
    p.add_argument('--out_dir', type=Path, default=Path('mmpose_raw8_20classes'))
    p.add_argument('--extract', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--mmpose_device', type=str, default='cpu')
    p.add_argument('--frame_stride', type=int, default=5)
    p.add_argument('--max_frames', type=int, default=60)
    p.add_argument('--minimum_detection_rate', type=float, default=0.3)
    p.add_argument('--segments', type=int, default=5)
    p.add_argument('--sampled_frames', type=int, default=10)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--only', nargs='+', choices=RUNS, default=RUNS)
    args = p.parse_args()
    manifest = load_manifest(args.manifest)
    if args.extract:
        extract_all(args, manifest)
    filtered, rejected = quality_filter(manifest, args.landmarks_dir, args.minimum_detection_rate)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rejected, columns=['sample_id', 'reason']).to_csv(args.out_dir / 'rejected_samples.csv', index=False)
    filtered.to_csv(args.out_dir / 'filtered_manifest.csv', index=False)
    print(f'[quality] kept={len(filtered)} rejected={len(rejected)}')
    Xtab, Xseq, y = build_features(filtered, args.landmarks_dir, args.max_frames, args.segments, args.sampled_frames, args.out_dir / 'feature_cache')
    results = {}
    for name in ['rf', 'svm', 'mlp', 'lstm']:
        run = f'mmpose_{name}'
        if run not in args.only:
            continue
        print(f'\n=== Running {run} ===')
        if name == 'lstm':
            results[run] = train_lstm(filtered, Xseq, y, args.out_dir / run, args.seed, args.epochs, args.batch_size)
        else:
            results[run] = train_classical(name, filtered, Xtab, y, args.out_dir / run, args.seed)
    summary = pd.DataFrame(results).T
    summary.to_csv(args.out_dir / 'mmpose_summary.csv')
    print('\n=== MMPOSE FINAL SUMMARY ===')
    print(summary)
    print(f'\nSaved to: {args.out_dir.resolve()}')
if __name__ == '__main__':
    main()
