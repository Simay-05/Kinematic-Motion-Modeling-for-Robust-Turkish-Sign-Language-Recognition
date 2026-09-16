from __future__ import annotations
import argparse, itertools, json, math, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
EPS = 1e-08

def safe_stem(value):
    text = str(value).strip().replace('\\', '/').split('/')[-1]
    for ext in ['.mp4', '.avi', '.mov', '.mkv', '.npy']:
        if text.lower().endswith(ext):
            text = text[:-len(ext)]
    return text

def load_manifest(path):
    df = pd.read_csv(path)
    df['sample_id'] = df['sample_id'].astype(str)
    df['label'] = df['label'].astype(str)
    df['split'] = df['split'].astype(str).str.lower()
    df.loc[df['split'].isin(['valid', 'validation']), 'split'] = 'val'
    return df.reset_index(drop=True)

def gaussian_kernel(sigma):
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    r = max(1, int(round(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(np.float32)

def smooth_time(x, sigma):
    k = gaussian_kernel(sigma)
    if len(k) == 1:
        return x.astype(np.float32)
    shp = x.shape
    flat = x.reshape(len(x), -1)
    r = len(k) // 2
    pad = np.pad(flat, ((r, r), (0, 0)), mode='edge')
    out = np.empty_like(flat, dtype=np.float32)
    for c in range(flat.shape[1]):
        out[:, c] = np.convolve(pad[:, c], k, mode='valid')
    return out.reshape(shp)

def resample(x, frames):
    if len(x) == frames:
        return x.astype(np.float32)
    if len(x) == 1:
        return np.repeat(x, frames, axis=0).astype(np.float32)
    old = np.linspace(0, 1, len(x))
    new = np.linspace(0, 1, frames)
    flat = x.reshape(len(x), -1)
    out = np.empty((frames, flat.shape[1]), dtype=np.float32)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(new, old, flat[:, c])
    return out.reshape((frames,) + x.shape[1:])

def normalize(seq, extractor):
    arr = np.asarray(seq, dtype=np.float32)
    arr[~np.isfinite(arr)] = 0
    if extractor == 'mediapipe':
        xyz = arr[:, :, :3].copy()
        out = np.zeros_like(xyz)
        for t, f in enumerate(xyz):
            center = f[[11, 12, 23, 24]].mean(0)
            scale = np.linalg.norm(f[11] - f[12])
            if not np.isfinite(scale) or scale < EPS:
                scale = 1.0
            out[t] = (f - center) / scale
        return out
    xy = arr[:, :, :2].copy()
    out2 = np.zeros_like(xy)
    for t, f in enumerate(xy):
        valid = np.linalg.norm(f, axis=1) > EPS
        anchors = [i for i in [5, 6, 11, 12] if valid[i]]
        if len(anchors) >= 2:
            center = f[anchors].mean(0)
        elif valid.any():
            center = f[valid].mean(0)
        else:
            out2[t] = out2[t - 1] if t else 0
            continue
        scale = np.linalg.norm(f[5] - f[6]) if valid[5] and valid[6] else 1.0
        if not np.isfinite(scale) or scale < EPS:
            scale = 1.0
        out2[t] = (f - center) / scale
        out2[t, ~valid] = 0
    out3 = np.zeros((len(out2), out2.shape[1], 3), dtype=np.float32)
    out3[:, :, :2] = out2
    return out3

def topology(extractor):
    if extractor == 'mediapipe':
        ls, rs = (33, 54)
        key = [11, 12, 13, 14, 15, 16, 33, 37, 41, 45, 49, 53, 54, 58, 62, 66, 70, 74]
        angles = {'left_elbow': (11, 13, 15), 'right_elbow': (12, 14, 16), 'left_shoulder': (13, 11, 23), 'right_shoulder': (14, 12, 24)}
    else:
        ls, rs = (17, 38)
        key = [5, 6, 7, 8, 9, 10, 17, 21, 25, 29, 33, 37, 38, 42, 46, 50, 54, 58]
        angles = {'left_elbow': (5, 7, 9), 'right_elbow': (6, 8, 10), 'left_shoulder': (7, 5, 11), 'right_shoulder': (8, 6, 12)}
    for side, s in [('left', ls), ('right', rs)]:
        angles.update({f'{side}_thumb_mcp': (s + 1, s + 2, s + 3), f'{side}_thumb_ip': (s + 2, s + 3, s + 4), f'{side}_index_mcp': (s + 5, s + 6, s + 7), f'{side}_index_pip': (s + 6, s + 7, s + 8), f'{side}_middle_mcp': (s + 9, s + 10, s + 11), f'{side}_middle_pip': (s + 10, s + 11, s + 12), f'{side}_ring_mcp': (s + 13, s + 14, s + 15), f'{side}_ring_pip': (s + 14, s + 15, s + 16), f'{side}_little_mcp': (s + 17, s + 18, s + 19), f'{side}_little_pip': (s + 18, s + 19, s + 20)})
    return (ls, rs, key, angles)

def angle_at_b(a, b, c):
    ba, bc = (a - b, c - b)
    d = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + EPS
    return np.arccos(np.clip(np.sum(ba * bc, axis=1) / d, -1, 1))

def entropy(sig, bins=10):
    if sig.ndim == 1:
        sig = sig[:, None]
    out = np.zeros(sig.shape[1], dtype=np.float32)
    for i in range(sig.shape[1]):
        x = sig[:, i]
        if np.allclose(x, x[0]):
            continue
        cnt, _ = np.histogram(x, bins=bins)
        p = cnt[cnt > 0].astype(float)
        p /= p.sum()
        out[i] = -np.sum(p * np.log(p + EPS)) / math.log(bins)
    return out

def curvature(pos):
    v = np.gradient(pos, axis=0)
    a = np.gradient(v, axis=0)
    k = np.linalg.norm(np.cross(v, a), axis=2) / (np.linalg.norm(v, axis=2) ** 3 + EPS)
    k[~np.isfinite(k)] = 0
    nz = k[k > 0]
    if nz.size:
        k = np.clip(k, 0, np.quantile(nz, 0.99))
    return k.astype(np.float32)

def summarize(sig, prefix, vals, names):
    if sig.ndim == 1:
        sig = sig[:, None]
    stats = {'mean': sig.mean(0), 'std': sig.std(0), 'min': sig.min(0), 'max': sig.max(0), 'median': np.median(sig, axis=0), 'q25': np.quantile(sig, 0.25, axis=0), 'q75': np.quantile(sig, 0.75, axis=0)}
    for st, x in stats.items():
        vals.append(x.astype(np.float32))
        names.extend([f'{prefix}_{st}_{i}' for i in range(sig.shape[1])])

def palm_orientation(hand):
    w, i, l = (hand[:, 0], hand[:, 5], hand[:, 17])
    n = np.cross(i - w, l - w)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + EPS
    az = np.arctan2(n[:, 1], n[:, 0])
    el = np.arctan2(n[:, 2], np.linalg.norm(n[:, :2], axis=1) + EPS)
    return np.concatenate([n, az[:, None], el[:, None]], axis=1)

def hand_shape(hand):
    w = hand[:, 0]
    tips = hand[:, [4, 8, 12, 16, 20]]
    mcps = hand[:, [2, 5, 9, 13, 17]]
    ps = np.linalg.norm(hand[:, 9] - w, axis=1)[:, None] + EPS
    adj = np.stack([np.linalg.norm(tips[:, i + 1] - tips[:, i], axis=1) for i in range(4)], axis=1) / ps
    th = np.stack([np.linalg.norm(tips[:, i] - tips[:, 0], axis=1) for i in range(1, 5)], axis=1) / ps
    tr = np.linalg.norm(tips - w[:, None, :], axis=2) / ps
    mr = np.linalg.norm(mcps - w[:, None, :], axis=2) / ps
    return (np.concatenate([adj, th], axis=1), np.concatenate([tr, mr, tr.mean(1, keepdims=True)], axis=1))

def physics_features(seq, extractor, frames, sigma):
    x = smooth_time(resample(normalize(seq, extractor), frames), sigma)
    ls, rs, key, angle_defs = topology(extractor)
    J = x.shape[1]
    v = np.gradient(x, axis=0)
    a = np.gradient(v, axis=0)
    j = np.gradient(a, axis=0)
    speed = np.linalg.norm(v, axis=2)
    accel = np.linalg.norm(a, axis=2)
    jerk = np.linalg.norm(j, axis=2)
    curv = curvature(x)
    vals = []
    names = []
    summarize(x[:, key].reshape(frames, -1), 'position', vals, names)
    summarize(speed, 'speed', vals, names)
    summarize(accel, 'acceleration', vals, names)
    summarize(curv, 'curvature', vals, names)
    path = speed.sum(0)
    disp = np.linalg.norm(x[-1] - x[0], axis=1)
    straight = disp / (path + EPS)
    energy = np.mean(speed ** 2, axis=0)
    smooth = np.sum(jerk ** 2, axis=0)
    for pfx, arr in [('path_length', path), ('displacement', disp), ('straightness', straight), ('motion_energy', energy), ('integrated_squared_jerk', smooth), ('speed_entropy', entropy(speed)), ('curvature_entropy', entropy(curv))]:
        vals.append(arr.astype(np.float32))
        names.extend([f'{pfx}_{i}' for i in range(J)])
    anames = list(angle_defs)
    ang = np.stack([angle_at_b(x[:, aa], x[:, bb], x[:, cc]) for aa, bb, cc in angle_defs.values()], axis=1)
    av = np.gradient(ang, axis=0)
    aa = np.gradient(av, axis=0)
    summarize(ang, 'joint_angle', vals, names)
    summarize(av, 'angular_velocity', vals, names)
    summarize(aa, 'angular_acceleration', vals, names)
    vals.append(entropy(av))
    names.extend([f'angular_entropy_{n}' for n in anames])
    left, right = (x[:, ls:ls + 21], x[:, rs:rs + 21])
    for side, hand in [('left', left), ('right', right)]:
        summarize(palm_orientation(hand), f'{side}_palm_orientation', vals, names)
        sp, op = hand_shape(hand)
        summarize(sp, f'{side}_finger_spread', vals, names)
        summarize(op, f'{side}_hand_openness', vals, names)
    bil = np.stack([np.linalg.norm(left[:, 0] - right[:, 0], axis=1), np.abs(speed[:, ls] - speed[:, rs]), np.mean(np.linalg.norm(left - right, axis=2), axis=1)], axis=1)
    summarize(bil, 'bilateral', vals, names)
    vec = np.concatenate(vals).astype(np.float32)
    vec[~np.isfinite(vec)] = 0
    return (vec, names)

def build_dataset(manifest, ldir, extractor, frames, sigma, cache):
    if cache.exists():
        d = joblib.load(cache)
        print(f'[cache:{extractor}] loaded')
        return (d['X'], d['y'], d['names'])
    X = []
    y = []
    names = None
    for i, row in manifest.iterrows():
        if i % 100 == 0:
            print(f'[features:{extractor}] {i}/{len(manifest)}')
        p = ldir / f"{safe_stem(row['sample_id'])}.npy"
        if not p.exists():
            raise FileNotFoundError(p)
        vec, nm = physics_features(np.load(p), extractor, frames, sigma)
        X.append(vec)
        y.append(str(row['label']))
        names = names or nm
    d = {'X': np.vstack(X), 'y': np.asarray(y), 'names': names}
    cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(d, cache)
    print(f"[features:{extractor}] shape={d['X'].shape}")
    return (d['X'], d['y'], d['names'])

def split_indices(manifest):
    s = manifest['split'].to_numpy()
    return (np.where(s == 'train')[0], np.where(s == 'val')[0], np.where(s == 'test')[0])

def signer_groups(manifest):
    if 'signer' in manifest.columns:
        return manifest['signer'].astype(str).to_numpy()
    return np.asarray([safe_stem(x).split('_sample')[0] for x in manifest['sample_id']])

def make_selector(name, val):
    if name == 'none':
        return 'passthrough'
    if name == 'kbest':
        return SelectKBest(mutual_info_classif, k=int(val))
    return Pipeline([('scale', StandardScaler()), ('pca', PCA(n_components=float(val), svd_solver='full'))])

def candidate_configs(nf):
    ks = sorted(set([min(250, nf), min(500, nf), min(1000, nf)]))
    sels = [('none', None)] + [('kbest', k) for k in ks] + [('pca', 0.9), ('pca', 0.95)]
    for sn, sv in sels:
        for trees, depth, leaf, mf in itertools.product([500, 800], [None, 30, 60], [1, 2, 4], ['sqrt', 0.2, 0.35]):
            yield {'selector_name': sn, 'selector_value': sv, 'rf': {'n_estimators': trees, 'max_depth': depth, 'min_samples_leaf': leaf, 'min_samples_split': 2, 'max_features': mf, 'bootstrap': True, 'class_weight': 'balanced_subsample', 'n_jobs': -1}}

def make_pipeline(cfg, seed):
    return Pipeline([('selector', make_selector(cfg['selector_name'], cfg['selector_value'])), ('rf', RandomForestClassifier(random_state=seed, **cfg['rf']))])

def cv_splits(y, groups, folds, seed):
    if len(set(groups)) >= folds:
        print(f'[cv] GroupKFold({folds})')
        return list(GroupKFold(folds).split(np.zeros(len(y)), y, groups))
    n = max(2, min(folds, int(pd.Series(y).value_counts().min())))
    return list(StratifiedKFold(n, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))

def tune(X, y, groups, folds, seed, maxc, out):
    splits = cv_splits(y, groups, folds, seed)
    opts = list(candidate_configs(X.shape[1]))
    if maxc and len(opts) > maxc:
        rng = np.random.default_rng(seed)
        ids = sorted(rng.choice(len(opts), maxc, replace=False))
        opts = [opts[i] for i in ids]
    best = None
    bf = -1
    bs = 1000000000.0
    ba = -1
    rec = []
    print(f'[search] {len(opts)} configs x {len(splits)} folds')
    for i, cfg in enumerate(opts, 1):
        fs = []
        ac = []
        for fold, (fit, chk) in enumerate(splits, 1):
            m = make_pipeline(cfg, seed + fold)
            m.fit(X[fit], y[fit])
            p = m.predict(X[chk])
            fs.append(f1_score(y[chk], p, average='macro', zero_division=0))
            ac.append(accuracy_score(y[chk], p))
        mf, sd, ma = (float(np.mean(fs)), float(np.std(fs)), float(np.mean(ac)))
        rec.append({'config': i, 'cv_macro_f1': mf, 'cv_std_f1': sd, 'cv_accuracy': ma, 'selector': cfg['selector_name'], 'selector_value': cfg['selector_value'], 'rf': json.dumps(cfg['rf'])})
        print(f"[search] {i:03d}/{len(opts)} f1={mf:.4f}±{sd:.4f} {cfg['selector_name']}:{cfg['selector_value']}")
        if mf > bf or (np.isclose(mf, bf) and sd < bs) or (np.isclose(mf, bf) and np.isclose(sd, bs) and (ma > ba)):
            best, bf, bs, ba = (cfg, mf, sd, ma)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rec).sort_values(['cv_macro_f1', 'cv_std_f1', 'cv_accuracy'], ascending=[False, True, False]).to_csv(out / 'cv_search.csv', index=False)
    return (best, bf, bs, ba)

def run_one(name, manifest, ldir, args):
    out = args.out_dir / name
    X, y, names = build_dataset(manifest, ldir, name, args.frames, args.smoothing_sigma, out / f'features_sigma{args.smoothing_sigma:.2f}.joblib')
    tr, val, te = split_indices(manifest)
    groups = signer_groups(manifest)
    cfg, cvf, cvs, cva = tune(X[tr], y[tr], groups[tr], args.cv_folds, args.seed, args.max_configurations, out)
    vm = make_pipeline(cfg, args.seed)
    vm.fit(X[tr], y[tr])
    vp = vm.predict(X[val])
    vf = float(f1_score(y[val], vp, average='macro', zero_division=0))
    va = float(accuracy_score(y[val], vp))
    fit = np.concatenate([tr, val])
    model = make_pipeline(cfg, args.seed)
    t = time.perf_counter()
    model.fit(X[fit], y[fit])
    train_s = time.perf_counter() - t
    t = time.perf_counter()
    pred = model.predict(X[te])
    infer_s = time.perf_counter() - t
    metrics = {'extractor': name, 'accuracy': float(accuracy_score(y[te], pred)), 'macro_f1': float(f1_score(y[te], pred, average='macro', zero_division=0)), 'weighted_f1': float(f1_score(y[te], pred, average='weighted', zero_division=0)), 'cv_macro_f1': cvf, 'cv_std_f1': cvs, 'cv_accuracy': cva, 'validation_macro_f1': vf, 'validation_accuracy': va, 'n_features_before_selection': int(X.shape[1]), 'training_seconds': train_s, 'inference_seconds': infer_s, 'best_configuration': cfg}
    joblib.dump(model, out / 'physics_rf.joblib')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    (out / 'classification_report.txt').write_text(classification_report(y[te], pred, zero_division=0))
    labels = sorted(set(y[te]) | set(pred))
    pd.DataFrame(confusion_matrix(y[te], pred, labels=labels), index=labels, columns=labels).to_csv(out / 'confusion_matrix.csv')
    return metrics

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--mediapipe_landmarks', type=Path, required=True)
    p.add_argument('--mmpose_landmarks', type=Path, required=True)
    p.add_argument('--out_dir', type=Path, default=Path('dual_kinematic_rf'))
    p.add_argument('--frames', type=int, default=60)
    p.add_argument('--smoothing_sigma', type=float, default=1.25)
    p.add_argument('--cv_folds', type=int, default=5)
    p.add_argument('--max_configurations', type=int, default=48)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--only', nargs='+', choices=['mediapipe', 'mmpose'], default=['mediapipe', 'mmpose'])
    args = p.parse_args()
    manifest = load_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dirs = {'mediapipe': args.mediapipe_landmarks, 'mmpose': args.mmpose_landmarks}
    results = {}
    for name in args.only:
        print('\n' + '=' * 70)
        print(f'KINEMATIC RF: {name.upper()}')
        print('=' * 70)
        results[name] = run_one(name, manifest, dirs[name], args)
    summary = pd.DataFrame(results).T
    summary.to_csv(args.out_dir / 'kinematic_rf_summary.csv')
    print('\n=== KINEMATIC RF COMPARISON ===')
    print(summary[['accuracy', 'macro_f1', 'weighted_f1', 'cv_macro_f1', 'validation_macro_f1', 'n_features_before_selection']])
    print(f'\nSaved to: {args.out_dir.resolve()}')
if __name__ == '__main__':
    main()
