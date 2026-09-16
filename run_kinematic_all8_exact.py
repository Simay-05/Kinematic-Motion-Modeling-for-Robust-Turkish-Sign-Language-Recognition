from __future__ import annotations
import argparse
import copy
import importlib.util
import itertools
import json
import random
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
HERE = Path(__file__).resolve().parent
EPS = 1e-08
ALL_RUNS = ['mediapipe_rf', 'mediapipe_svm', 'mediapipe_mlp', 'mediapipe_bilstm', 'mmpose_rf', 'mmpose_svm', 'mmpose_mlp', 'mmpose_bilstm']

def load_dual():
    path = HERE / 'run_dual_kinematic_rf.py'
    if not path.exists():
        raise FileNotFoundError('run_dual_kinematic_rf.py must be in the same folder.')
    spec = importlib.util.spec_from_file_location('dual_kin', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules['dual_kin'] = module
    spec.loader.exec_module(module)
    return module
dual = load_dual()

def split_indices(manifest):
    s = manifest['split'].to_numpy()
    return (np.where(s == 'train')[0], np.where(s == 'val')[0], np.where(s == 'test')[0])

def make_rf(params, seed):
    return RandomForestClassifier(random_state=seed, n_jobs=-1, class_weight='balanced_subsample', bootstrap=True, **params)

def tune_rf(Xtr, ytr, Xv, yv, seed, out):
    configs = []
    for trees, depth, leaf, mf in itertools.product([500, 800], [None, 30, 60], [1, 2, 4], ['sqrt', 0.2, 0.35]):
        configs.append({'n_estimators': trees, 'max_depth': depth, 'min_samples_leaf': leaf, 'min_samples_split': 2, 'max_features': mf})
    best = None
    best_f1 = -1.0
    best_acc = -1.0
    records = []
    for i, params in enumerate(configs, 1):
        model = make_rf(params, seed)
        model.fit(Xtr, ytr)
        pred = model.predict(Xv)
        f1 = f1_score(yv, pred, average='macro', zero_division=0)
        acc = accuracy_score(yv, pred)
        records.append({'configuration': i, 'validation_macro_f1': f1, 'validation_accuracy': acc, 'parameters': json.dumps(params)})
        print(f'[rf] {i:02d}/{len(configs)} f1={f1:.4f} acc={acc:.4f}')
        if f1 > best_f1 or (np.isclose(f1, best_f1) and acc > best_acc):
            best = params
            best_f1 = float(f1)
            best_acc = float(acc)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).sort_values(['validation_macro_f1', 'validation_accuracy'], ascending=False).to_csv(out / 'validation_search.csv', index=False)
    return (best, best_f1, best_acc)

def make_svm(params):
    return Pipeline([('scale', StandardScaler()), ('pca', PCA(n_components=params['pca_variance'], svd_solver='full')), ('model', SVC(kernel='rbf', C=params['C'], gamma=params['gamma'], class_weight='balanced'))])

def tune_svm(Xtr, ytr, Xv, yv, out):
    configs = [{'pca_variance': variance, 'C': C, 'gamma': gamma} for variance, C, gamma in itertools.product([0.9, 0.95, 0.99], [0.3, 1.0, 3.0, 10.0, 30.0, 100.0], ['scale', 0.0003, 0.001, 0.003, 0.01])]
    best = None
    best_f1 = -1.0
    best_acc = -1.0
    records = []
    for i, params in enumerate(configs, 1):
        model = make_svm(params)
        model.fit(Xtr, ytr)
        pred = model.predict(Xv)
        f1 = f1_score(yv, pred, average='macro', zero_division=0)
        acc = accuracy_score(yv, pred)
        records.append({'configuration': i, 'validation_macro_f1': f1, 'validation_accuracy': acc, 'parameters': json.dumps(params)})
        print(f'[svm] {i:03d}/{len(configs)} f1={f1:.4f} acc={acc:.4f}')
        if f1 > best_f1 or (np.isclose(f1, best_f1) and acc > best_acc):
            best = params
            best_f1 = float(f1)
            best_acc = float(acc)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).sort_values(['validation_macro_f1', 'validation_accuracy'], ascending=False).to_csv(out / 'validation_search.csv', index=False)
    return (best, best_f1, best_acc)

def make_mlp(params, seed):
    return Pipeline([('scale', StandardScaler()), ('pca', PCA(n_components=params['pca_variance'], svd_solver='full')), ('model', MLPClassifier(hidden_layer_sizes=tuple(params['hidden_layers']), activation='relu', solver='adam', alpha=params['alpha'], batch_size=32, learning_rate_init=params['learning_rate'], early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, max_iter=500, random_state=seed))])

def tune_mlp(Xtr, ytr, Xv, yv, seed, out):
    configs = [{'pca_variance': variance, 'hidden_layers': layers, 'alpha': alpha, 'learning_rate': lr} for variance, layers, alpha, lr in itertools.product([0.9, 0.95], [(256,), (256, 128), (512, 256)], [0.0001, 0.001], [0.0005, 0.001])]
    best = None
    best_f1 = -1.0
    best_acc = -1.0
    records = []
    for i, params in enumerate(configs, 1):
        model = make_mlp(params, seed)
        model.fit(Xtr, ytr)
        pred = model.predict(Xv)
        f1 = f1_score(yv, pred, average='macro', zero_division=0)
        acc = accuracy_score(yv, pred)
        records.append({'configuration': i, 'validation_macro_f1': f1, 'validation_accuracy': acc, 'parameters': json.dumps(params)})
        print(f'[mlp] {i:02d}/{len(configs)} f1={f1:.4f} acc={acc:.4f}')
        if f1 > best_f1 or (np.isclose(f1, best_f1) and acc > best_acc):
            best = params
            best_f1 = float(f1)
            best_acc = float(acc)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).sort_values(['validation_macro_f1', 'validation_accuracy'], ascending=False).to_csv(out / 'validation_search.csv', index=False)
    return (best, best_f1, best_acc)

def save_classical_results(extractor, model_name, X, y, manifest, out, seed):
    train, val, test = split_indices(manifest)
    if model_name == 'rf':
        params, val_f1, val_acc = tune_rf(X[train], y[train], X[val], y[val], seed, out)
        final_model = make_rf(params, seed)
    elif model_name == 'svm':
        params, val_f1, val_acc = tune_svm(X[train], y[train], X[val], y[val], out)
        final_model = make_svm(params)
    elif model_name == 'mlp':
        params, val_f1, val_acc = tune_mlp(X[train], y[train], X[val], y[val], seed, out)
        final_model = make_mlp(params, seed)
    else:
        raise ValueError(model_name)
    fit = np.concatenate([train, val])
    t0 = time.perf_counter()
    final_model.fit(X[fit], y[fit])
    training_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = final_model.predict(X[test])
    inference_seconds = time.perf_counter() - t0
    joblib.dump(final_model, out / 'model.joblib')
    metrics = {'extractor': extractor, 'model': model_name, 'feature_mode': 'exact_rf_physics_vector', 'accuracy': float(accuracy_score(y[test], pred)), 'macro_f1': float(f1_score(y[test], pred, average='macro', zero_division=0)), 'weighted_f1': float(f1_score(y[test], pred, average='weighted', zero_division=0)), 'validation_macro_f1': val_f1, 'validation_accuracy': val_acc, 'n_features': int(X.shape[1]), 'best_parameters': params, 'training_seconds': training_seconds, 'inference_seconds': inference_seconds}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    (out / 'classification_report.txt').write_text(classification_report(y[test], pred, zero_division=0), encoding='utf-8')
    labels = sorted(set(y[test]) | set(pred))
    pd.DataFrame(confusion_matrix(y[test], pred, labels=labels), index=labels, columns=labels).to_csv(out / 'confusion_matrix.csv')
    return metrics

def global_sequence_descriptors(x, speed, accel, jerk, curv):
    path = speed.sum(axis=0)
    disp = np.linalg.norm(x[-1] - x[0], axis=1)
    straight = disp / (path + EPS)
    energy = np.mean(speed ** 2, axis=0)
    smoothness = np.sum(jerk ** 2, axis=0)
    speed_entropy = dual.entropy(speed)
    curvature_entropy = dual.entropy(curv)
    return np.concatenate([path, disp, straight, energy, smoothness, speed_entropy, curvature_entropy]).astype(np.float32)

def kinematic_sequence(raw_sequence, extractor, frames, sigma):
    x = dual.normalize(raw_sequence, extractor)
    x = dual.resample(x, frames)
    x = dual.smooth_time(x, sigma)
    ls, rs, key_indices, angle_defs = dual.topology(extractor)
    if extractor == 'mmpose':
        pos = x[:, :, :2]
    else:
        pos = x[:, :, :3]
    vel = np.gradient(pos, axis=0)
    acc = np.gradient(vel, axis=0)
    jerk_vec = np.gradient(acc, axis=0)
    speed = np.linalg.norm(vel, axis=2)
    accel_mag = np.linalg.norm(acc, axis=2)
    jerk_mag = np.linalg.norm(jerk_vec, axis=2)
    x_for_curv = x[:, :, :3]
    curv = dual.curvature(x_for_curv)
    angles = np.stack([dual.angle_at_b(x[:, a], x[:, b], x[:, c]) for a, b, c in angle_defs.values()], axis=1).astype(np.float32)
    angular_v = np.gradient(angles, axis=0)
    angular_a = np.gradient(angular_v, axis=0)
    left = x[:, ls:ls + 21]
    right = x[:, rs:rs + 21]
    left_orientation = dual.palm_orientation(left)
    right_orientation = dual.palm_orientation(right)
    left_spread, left_open = dual.hand_shape(left)
    right_spread, right_open = dual.hand_shape(right)
    bilateral = np.stack([np.linalg.norm(left[:, 0] - right[:, 0], axis=1), np.abs(speed[:, ls] - speed[:, rs]), np.mean(np.linalg.norm(left - right, axis=2), axis=1)], axis=1)
    global_desc = global_sequence_descriptors(x_for_curv, speed, accel_mag, jerk_mag, curv)
    repeated_global = np.repeat(global_desc[None, :], frames, axis=0)
    key_positions = pos[:, key_indices].reshape(frames, -1)
    features = np.concatenate([key_positions, vel.reshape(frames, -1), acc.reshape(frames, -1), speed, accel_mag, curv, angles, angular_v, angular_a, left_orientation, right_orientation, left_spread, right_spread, left_open, right_open, bilateral, repeated_global], axis=1).astype(np.float32)
    features[~np.isfinite(features)] = 0.0
    return features

def build_sequence_dataset(manifest, landmark_dir, extractor, frames, sigma, cache_path):
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        print(f'[cache:{extractor}:sequence] loaded')
        return (data['X'], data['y'])
    X = []
    y = []
    for i, row in manifest.iterrows():
        if i % 100 == 0:
            print(f'[sequence:{extractor}] {i}/{len(manifest)}')
        path = landmark_dir / f"{dual.safe_stem(row['sample_id'])}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        X.append(kinematic_sequence(np.load(path), extractor, frames, sigma))
        y.append(str(row['label']))
    X = np.stack(X)
    y = np.asarray(y)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y)
    print(f'[sequence:{extractor}] shape={X.shape}')
    return (X, y)

class KinematicBiLSTM(nn.Module):

    def __init__(self, input_size, class_count, hidden_size=160, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.projection = nn.Sequential(nn.Linear(input_size, 256), nn.GELU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = nn.Sequential(nn.Linear(hidden_size * 2, 128), nn.Tanh(), nn.Linear(128, 1))
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_size * 2), nn.Dropout(dropout), nn.Linear(hidden_size * 2, class_count))

    def forward(self, x):
        x = self.projection(self.norm(x))
        seq, _ = self.lstm(x)
        scores = self.attention(seq).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(seq * weights.unsqueeze(-1), dim=1)
        return self.classifier(pooled)

def choose_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def train_bilstm(extractor, X, y, manifest, out, args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train, val, test = split_indices(manifest)
    encoder = LabelEncoder()
    encoder.fit(y[train])
    encoded = encoder.transform(y)
    scaler = StandardScaler()
    scaler.fit(X[train].reshape(-1, X.shape[-1]))

    def scaled(indices):
        shape = X[indices].shape
        return scaler.transform(X[indices].reshape(-1, X.shape[-1])).reshape(shape).astype(np.float32)
    train_ds = TensorDataset(torch.tensor(scaled(train)), torch.tensor(encoded[train], dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(scaled(val)), torch.tensor(encoded[val], dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(scaled(test)), torch.tensor(encoded[test], dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    device = choose_device()
    model = KinematicBiLSTM(X.shape[-1], len(encoder.classes_)).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0003)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-05)
    best_state = None
    best_f1 = -1.0
    best_loss = float('inf')
    best_acc = -1.0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = loss_fn(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(train_ds)
        model.eval()
        validation_loss = 0.0
        vp = []
        vt = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                by = by.to(device)
                logits = model(bx)
                validation_loss += loss_fn(logits, by).item() * len(bx)
                vp.extend(logits.argmax(dim=1).cpu().numpy())
                vt.extend(by.cpu().numpy())
        validation_loss /= len(val_ds)
        validation_f1 = f1_score(vt, vp, average='macro', zero_division=0)
        validation_accuracy = accuracy_score(vt, vp)
        scheduler.step(validation_loss)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'validation_loss': validation_loss, 'validation_macro_f1': float(validation_f1), 'validation_accuracy': float(validation_accuracy), 'learning_rate': optimizer.param_groups[0]['lr']})
        print(f'[{extractor}+BiLSTM] epoch={epoch:03d} train={train_loss:.4f} val={validation_loss:.4f} f1={validation_f1:.4f} acc={validation_accuracy:.4f}')
        improved = validation_f1 > best_f1 + 0.0001 or (np.isclose(validation_f1, best_f1) and validation_loss < best_loss)
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_f1 = float(validation_f1)
            best_loss = float(validation_loss)
            best_acc = float(validation_accuracy)
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print('[early stopping]')
                break
    if best_state is None:
        raise RuntimeError('No valid BiLSTM state was produced.')
    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for bx, _ in test_loader:
            logits = model(bx.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().numpy())
    inference_seconds = time.perf_counter() - t0
    pred = encoder.inverse_transform(np.asarray(predictions))
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / 'kinematic_bilstm.pt')
    joblib.dump(scaler, out / 'scaler.joblib')
    joblib.dump(encoder, out / 'label_encoder.joblib')
    (out / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    metrics = {'extractor': extractor, 'model': 'bilstm', 'feature_mode': 'same_physics_pipeline_sequentialized', 'accuracy': float(accuracy_score(y[test], pred)), 'macro_f1': float(f1_score(y[test], pred, average='macro', zero_division=0)), 'weighted_f1': float(f1_score(y[test], pred, average='weighted', zero_division=0)), 'validation_macro_f1': best_f1, 'validation_accuracy': best_acc, 'best_validation_loss': best_loss, 'sequence_frames': int(X.shape[1]), 'features_per_frame': int(X.shape[2]), 'device': str(device), 'inference_seconds': float(inference_seconds)}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    (out / 'classification_report.txt').write_text(classification_report(y[test], pred, zero_division=0), encoding='utf-8')
    labels = sorted(set(y[test]) | set(pred))
    pd.DataFrame(confusion_matrix(y[test], pred, labels=labels), index=labels, columns=labels).to_csv(out / 'confusion_matrix.csv')
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--mediapipe_landmarks', type=Path, required=True)
    parser.add_argument('--mmpose_landmarks', type=Path, required=True)
    parser.add_argument('--out_dir', type=Path, default=Path('kinematic_all8'))
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--smoothing_sigma', type=float, default=1.25)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0007)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--only', nargs='+', choices=ALL_RUNS, default=ALL_RUNS)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = dual.load_manifest(args.manifest)
    landmark_dirs = {'mediapipe': args.mediapipe_landmarks, 'mmpose': args.mmpose_landmarks}
    results = {}
    for extractor in ['mediapipe', 'mmpose']:
        requested = [name for name in args.only if name.startswith(extractor + '_')]
        if not requested:
            continue
        vector_cache = args.out_dir / extractor / f'exact_rf_physics_sigma{args.smoothing_sigma:.2f}.joblib'
        X_tabular, y, _ = dual.build_dataset(manifest, landmark_dirs[extractor], extractor, args.frames, args.smoothing_sigma, vector_cache)
        for model_name in ['rf', 'svm', 'mlp']:
            run_name = f'{extractor}_{model_name}'
            if run_name not in args.only:
                continue
            print('\n' + '=' * 72)
            print(run_name.upper() + ' + KINEMATICS')
            print('=' * 72)
            results[run_name] = save_classical_results(extractor, model_name, X_tabular, y, manifest, args.out_dir / run_name, args.seed)
        bilstm_name = f'{extractor}_bilstm'
        if bilstm_name in args.only:
            print('\n' + '=' * 72)
            print(bilstm_name.upper() + ' + KINEMATICS')
            print('=' * 72)
            sequence_cache = args.out_dir / extractor / f'sequential_physics_sigma{args.smoothing_sigma:.2f}.npz'
            X_seq, y_seq = build_sequence_dataset(manifest, landmark_dirs[extractor], extractor, args.frames, args.smoothing_sigma, sequence_cache)
            results[bilstm_name] = train_bilstm(extractor, X_seq, y_seq, manifest, args.out_dir / bilstm_name, args)
    summary = pd.DataFrame(results).T
    summary.to_csv(args.out_dir / 'kinematic_all8_summary.csv')
    print('\n=== KINEMATIC ALL-8 SUMMARY ===')
    display = [c for c in ['accuracy', 'macro_f1', 'weighted_f1', 'validation_macro_f1'] if c in summary.columns]
    print(summary[display])
    print(f'\nSaved to: {args.out_dir.resolve()}')
if __name__ == '__main__':
    main()
