from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
HERE = Path(__file__).resolve().parent

def load_base():
    path = HERE / 'autsl_base.py'
    spec = importlib.util.spec_from_file_location('autsl_base', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules['autsl_base'] = module
    spec.loader.exec_module(module)
    return module
base = load_base()
EPS = 1e-08

def balanced_rows(rows, n_classes, train_per_class, val_per_class, test_per_class, seed):
    rng = random.Random(seed)
    by_key = {}
    for row in rows:
        split = str(row.split or '').lower()
        if split in {'valid', 'validation'}:
            split = 'val'
        by_key.setdefault((str(row.label), split), []).append(row)
    labels = sorted({str(r.label) for r in rows}, key=lambda x: (len(x), x))
    eligible = [label for label in labels if len(by_key.get((label, 'train'), [])) >= train_per_class and len(by_key.get((label, 'val'), [])) >= val_per_class and (len(by_key.get((label, 'test'), [])) >= test_per_class)]
    if len(eligible) < n_classes:
        raise ValueError(f'Only {len(eligible)} classes have enough samples; requested {n_classes}.')
    rng.shuffle(eligible)
    selected_labels = sorted(eligible[:n_classes], key=lambda x: (len(x), x))
    selected = []
    for label in selected_labels:
        for split, limit in (('train', train_per_class), ('val', val_per_class), ('test', test_per_class)):
            candidates = by_key[label, split][:]
            rng.shuffle(candidates)
            selected.extend(candidates[:limit])
    print(f'[sampling] {n_classes} classes; {train_per_class} train + {val_per_class} val + {test_per_class} test/class; total={len(selected)}')
    print(f'[sampling] labels={selected_labels}')
    return selected

def normalize(seq):
    xyz = np.asarray(seq, dtype=np.float32)[:, :, :3].copy()
    xyz[~np.isfinite(xyz)] = 0.0
    out = np.zeros_like(xyz)
    for t in range(len(xyz)):
        frame = xyz[t]
        valid = np.linalg.norm(frame[:, :2], axis=1) > EPS
        if valid.sum() < 5:
            out[t] = out[t - 1] if t > 0 else 0.0
            continue
        anchors = [i for i in [11, 12, 23, 24] if i < len(valid) and valid[i]]
        center = frame[anchors].mean(axis=0) if len(anchors) >= 2 else frame[valid].mean(axis=0)
        centered = frame - center
        if valid[11] and valid[12]:
            scale = np.linalg.norm(frame[11] - frame[12])
        else:
            scale = np.median(np.linalg.norm(centered[valid], axis=1))
        if not np.isfinite(scale) or scale < EPS:
            scale = 1.0
        out[t] = centered / scale
        out[t, ~valid] = 0.0
    return out

def resample(sequence, target_frames):
    if len(sequence) == target_frames:
        return sequence.astype(np.float32)
    if len(sequence) == 1:
        return np.repeat(sequence, target_frames, axis=0).astype(np.float32)
    old = np.linspace(0.0, 1.0, len(sequence))
    new = np.linspace(0.0, 1.0, target_frames)
    flat = sequence.reshape(len(sequence), -1)
    result = np.empty((target_frames, flat.shape[1]), dtype=np.float32)
    for column in range(flat.shape[1]):
        result[:, column] = np.interp(new, old, flat[:, column])
    return result.reshape((target_frames,) + sequence.shape[1:])

def raw_sequence(seq, frames):
    return resample(normalize(seq), frames).reshape(frames, -1).astype(np.float32)

def raw_tabular(sequence):
    return np.concatenate([np.mean(sequence, axis=0), np.std(sequence, axis=0), np.min(sequence, axis=0), np.max(sequence, axis=0), np.median(sequence, axis=0)]).astype(np.float32)

def build_features(rows, landmarks_dir, frames, cache_prefix):
    tab_cache = cache_prefix.with_name(cache_prefix.name + '_tab.joblib')
    seq_cache = cache_prefix.with_name(cache_prefix.name + '_seq.npz')
    if tab_cache.exists() and seq_cache.exists():
        tab = joblib.load(tab_cache)
        seq = np.load(seq_cache, allow_pickle=False)
        return (tab['X'], seq['X'], tab['y'])
    X_tab, X_seq, y = ([], [], [])
    for i, row in enumerate(rows):
        if i % 250 == 0:
            print(f'[features] {i}/{len(rows)}')
        path = landmarks_dir / f'{base.safe_stem(row.sample_id)}.npy'
        seq = np.load(path).astype(np.float32)
        seq_features = raw_sequence(seq, frames)
        X_seq.append(seq_features)
        X_tab.append(raw_tabular(seq_features))
        y.append(str(row.label))
    X_tab = np.vstack(X_tab)
    X_seq = np.stack(X_seq)
    y = np.asarray(y)
    joblib.dump({'X': X_tab, 'y': y}, tab_cache)
    np.savez_compressed(seq_cache, X=X_seq, y=y)
    return (X_tab, X_seq, y)

def split_indices(rows):
    splits = np.asarray([str(r.split).lower() for r in rows])
    train = np.where(splits == 'train')[0]
    val = np.where(np.isin(splits, ['val', 'valid', 'validation']))[0]
    test = np.where(splits == 'test')[0]
    return (train, val, test)

def create_classical(name, seed, trees):
    if name == 'rf':
        return RandomForestClassifier(n_estimators=trees, max_features='sqrt', class_weight='balanced_subsample', n_jobs=-1, random_state=seed)
    if name == 'svm':
        return Pipeline([('scale', StandardScaler()), ('model', SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced', probability=True, random_state=seed))])
    if name == 'mlp':
        return Pipeline([('scale', StandardScaler()), ('model', MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', early_stopping=True, validation_fraction=0.15, max_iter=350, random_state=seed))])
    raise ValueError(name)

class SignLSTM(nn.Module):

    def __init__(self, input_size, classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.head = nn.Sequential(nn.LayerNorm(256), nn.Dropout(0.3), nn.Linear(256, classes))

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1])

def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def save_metrics(y_true, y_pred, output, extras):
    output.mkdir(parents=True, exist_ok=True)
    metrics = {'accuracy': float(accuracy_score(y_true, y_pred)), 'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), **extras}
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    (output / 'classification_report.txt').write_text(classification_report(y_true, y_pred, zero_division=0))
    labels = sorted(set(y_true) | set(y_pred), key=str)
    pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels).to_csv(output / 'confusion_matrix.csv')
    return metrics

def train_classical(extractor, model_name, rows, X, y, output, seed, trees):
    train, val, test = split_indices(rows)
    fit = np.concatenate([train, val])
    model = create_classical(model_name, seed, trees)
    start = time.perf_counter()
    model.fit(X[fit], y[fit])
    training = time.perf_counter() - start
    start = time.perf_counter()
    pred = model.predict(X[test])
    inference = time.perf_counter() - start
    joblib.dump(model, output / 'model.joblib')
    return save_metrics(y[test], pred, output, {'extractor': extractor, 'model': model_name, 'feature_mode': 'raw_only', 'n_train_plus_val': int(len(fit)), 'n_test': int(len(test)), 'n_features': int(X.shape[1]), 'training_seconds': training, 'inference_seconds': inference})

def train_lstm(extractor, rows, X, y, output, seed, epochs, batch_size):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train, val, test = split_indices(rows)
    encoder = LabelEncoder()
    encoder.fit(y[train])
    encoded = encoder.transform(y)
    scaler = StandardScaler()
    scaler.fit(X[train].reshape(-1, X.shape[-1]))

    def scaled(idx):
        return scaler.transform(X[idx].reshape(-1, X.shape[-1])).reshape(X[idx].shape).astype(np.float32)
    train_data = TensorDataset(torch.tensor(scaled(train)), torch.tensor(encoded[train], dtype=torch.long))
    val_data = TensorDataset(torch.tensor(scaled(val)), torch.tensor(encoded[val], dtype=torch.long))
    test_data = TensorDataset(torch.tensor(scaled(test)), torch.tensor(encoded[test], dtype=torch.long))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    run_device = device()
    model = SignLSTM(X.shape[-1], len(encoder.classes_)).to(run_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_val = float('inf')
    stale = 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = (batch_x.to(run_device), batch_y.to(run_device))
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = (batch_x.to(run_device), batch_y.to(run_device))
                val_loss += criterion(model(batch_x), batch_y).item() * len(batch_x)
        val_loss /= max(len(val_data), 1)
        print(f'[{extractor}+lstm] epoch={epoch:03d} val={val_loss:.4f}')
        if val_loss < best_val - 0.0001:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= 10:
                break
    training = time.perf_counter() - start
    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    start = time.perf_counter()
    with torch.no_grad():
        for batch_x, _ in test_loader:
            predictions.extend(model(batch_x.to(run_device)).argmax(dim=1).cpu().numpy())
    inference = time.perf_counter() - start
    predicted = encoder.inverse_transform(np.asarray(predictions))
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / 'model.pt')
    joblib.dump(scaler, output / 'scaler.joblib')
    joblib.dump(encoder, output / 'label_encoder.joblib')
    return save_metrics(y[test], predicted, output, {'extractor': extractor, 'model': 'lstm', 'feature_mode': 'raw_only', 'n_train': int(len(train)), 'n_val': int(len(val)), 'n_test': int(len(test)), 'sequence_frames': int(X.shape[1]), 'features_per_frame': int(X.shape[2]), 'device': str(run_device), 'best_validation_loss': float(best_val), 'training_seconds': training, 'inference_seconds': inference})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--out_dir', type=Path, default=Path('raw_baseline_outputs'))
    parser.add_argument('--n_classes', type=int, default=20)
    parser.add_argument('--train_per_class', type=int, default=18)
    parser.add_argument('--val_per_class', type=int, default=6)
    parser.add_argument('--test_per_class', type=int, default=6)
    parser.add_argument('--frame_stride', type=int, default=5)
    parser.add_argument('--max_frames', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trees', type=int, default=400)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--skip_extract', action='store_true')
    parser.add_argument('--only', nargs='+', choices=['mediapipe_rf', 'mediapipe_svm', 'mediapipe_mlp', 'mediapipe_lstm'], default=None)
    args = parser.parse_args()
    requested = set(args.only or ['mediapipe_rf', 'mediapipe_svm', 'mediapipe_mlp', 'mediapipe_lstm'])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = balanced_rows(base.find_label_rows(args.data_dir, None), args.n_classes, args.train_per_class, args.val_per_class, args.test_per_class, args.seed)
    videos = base.find_video_files(args.data_dir, None)
    landmarks_dir = args.out_dir / 'mediapipe_landmarks'
    if not args.skip_extract:
        base.extract_landmarks_dataset(rows, videos, landmarks_dir, None, args.frame_stride, args.max_frames, False)
    X_tab, X_seq, y = build_features(rows, landmarks_dir, args.max_frames, args.out_dir / 'raw_features')
    results = {}
    for model_name in ['rf', 'svm', 'mlp', 'lstm']:
        run_name = f'mediapipe_{model_name}'
        if run_name not in requested:
            continue
        print(f'\n=== Running {run_name} ===')
        output = args.out_dir / run_name
        output.mkdir(parents=True, exist_ok=True)
        if model_name == 'lstm':
            results[run_name] = train_lstm('mediapipe', rows, X_seq, y, output, args.seed, args.epochs, args.batch_size)
        else:
            results[run_name] = train_classical('mediapipe', model_name, rows, X_tab, y, output, args.seed, args.trees)
    summary = pd.DataFrame(results).T
    summary.to_csv(args.out_dir / 'raw_baseline_summary.csv')
    print('\n=== FINAL SUMMARY ===')
    print(summary)
    print(f'\nSaved to: {args.out_dir.resolve()}')
if __name__ == '__main__':
    main()
