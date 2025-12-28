#!/usr/bin/env python3
"""
DOGI Model Trainer
Simplified neural network trainer for storage access pattern prediction.
Matches mlp_inference.cc C++ implementation.

Training dataset format: idx, lba, prevlba, interval, freq, freq2, seg_accessed, real_interval
Feature order must match mlp_inference.h:
  [LBA, freq_bit, freq_bit2, interval_bit, seg_accessed, prev_lba]
"""

import os

# =============================================================================
# Force SINGLE-CORE execution (process + native libs)
# =============================================================================
# NOTE: Many numeric libs (OpenMP/MKL/OpenBLAS/BLIS/NumExpr) spin up worker threads.
# These must be constrained *before* importing numpy/pandas/torch/sklearn.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OMP_DYNAMIC'] = 'FALSE'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Pin the entire process to a single CPU core (Linux).
# This is the strongest guarantee: even if libraries try to spawn threads,
# they can only run on that one core.
try:
    os.sched_setaffinity(0, {0})
except Exception:
    # Not available on some platforms or restricted by container/cgroup.
    pass

import re
import sys
import contextlib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Force CPU and single-threaded execution inside PyTorch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


@contextlib.contextmanager
def _maybe_suppress_stdout(suppress: bool):
    """Optionally suppress stdout (keeps stderr for errors/tracebacks)."""
    if not suppress:
        yield
        return
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


# =============================================================================
# Configuration (matching mlp_inference.h)
# =============================================================================

# Default hyperparameters
DEFAULT_LR = 0.0003
DEFAULT_WEIGHT_DECAY = 0.00001
DEFAULT_DROPOUT_RATE = 0.4
DEFAULT_NUM_EPOCHS = 15
DEFAULT_BATCH_SIZE = 128
_PPS=16384.0
#_PPS=65536.0

# Fixed thresholds for 10 groups (default values)
DEFAULT_THRESHOLDS = np.array([1000.0, 5000.0, 10000.0, 20000.0, 50000.0, 
                                100000.0, 200000.0, 500000.0, 1000000.0], dtype=np.float32)
# Previous threshold sets (for Evaluation)
DEFAULT_THRESHOLDS2 = np.array([517857.0, 2578585.0, 5837334.0, 9630480.0, 
                                21229424.0, 39172404.0, 66404444.0, 110526010.0, 198227020.0], dtype=np.float32)
# threshold sets (written in paper)
DEFAULT_THRESHOLDS3 = np.array([335544.0, 671088.0, 1342176.0, 2684352.0, 
                                5368704.0, 10737408.0, 21474816.0, 42949632.0, 85899345.0], dtype=np.float32)

# Will be computed at runtime from loaded samples (balanced / equal-frequency groups)
DEFAULT_THRESHOLDS4 = np.array([], dtype=np.float32)

# Input features (6 features) - ORDER MUST MATCH mlp_inference.h
# [LBA, freq_bit, freq_bit2, interval_bit, seg_accessed, prev_lba]
INPUT_FEATURES = ['LBA', 'FreqBits', 'FreqBits2', 'Interval', 'SegFreq', 'PrevLBA']
TARGET_FEATURE = 'RealInterval'

# Model architecture - MUST MATCH mlp_inference.h
INPUT_DIM = 6
HIDDEN_LAYERS = [32, 32]  # kHidden1=32, kHidden2=32
NUM_GROUPS = 10           # kOutputDim=10

# FOR DEBUGGING
LARGE_INTERV = 13662825.0

# =============================================================================
# Dynamic Neural Network Model
# =============================================================================

class DynamicNN(nn.Module):
    """
    Dynamic Multi-Layer Perceptron with configurable hidden layers.
    
    Args:
        input_size: Number of input features (default: 6)
        num_classes: Number of output classes (groups)
        hidden_layers: List of hidden layer sizes (e.g., [32, 32])
        dropout_rate: Dropout probability (default: 0.4)
    """
    def __init__(self, input_size, num_classes, hidden_layers, dropout_rate=0.4):
        super(DynamicNN, self).__init__()
        layers = []
        in_features = input_size
        
        # Build hidden layers
        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = h
        
        # Output layer
        layers.append(nn.Linear(in_features, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


# =============================================================================
# Data Preparation
# =============================================================================

def compute_balanced_thresholds(real_interval: np.ndarray, num_groups: int) -> np.ndarray:
    """Compute thresholds so each digitized group has ~equal sample count.

    Returns an array of length (num_groups-1) to be used with np.digitize.
    Filters out NaN and sentinel -1 values before computing.
    """
    values = np.asarray(real_interval, dtype=np.float32)
    valid_mask = np.isfinite(values) & (values != -1)
    values = values[valid_mask]

    probs = np.linspace(0.0, 1.0, num_groups + 1, dtype=np.float64)[1:-1]
    thresholds = np.quantile(values.astype(np.float64), probs, method='linear').astype(np.float32)

    # Ensure strictly increasing thresholds to avoid degenerate bins when many values are identical.
    for i in range(1, thresholds.size):
        if not np.isfinite(thresholds[i]):
            thresholds[i] = thresholds[i - 1]
        if thresholds[i] <= thresholds[i - 1]:
            thresholds[i] = np.nextafter(thresholds[i - 1], np.float32(np.inf))

    return thresholds


def _format_float_array(arr: np.ndarray, precision: int) -> str:
    """Format a numeric array without scientific notation (no e+xx)."""
    a = np.asarray(arr)
    fmt = f"{{:.{precision}f}}".format
    return np.array2string(
        a,
        separator=', ',
        formatter={'float_kind': lambda x: fmt(float(x))},
        max_line_width=200000,
    )


def save_validation_predictions(
    output_dir: str,
    model_suffix: str,
    idx_val: np.ndarray,
    y_val: np.ndarray,
    real_interval_val: np.ndarray,
    pred_val: np.ndarray,
):
    """Save validation predictions sorted by Idx.

    Output CSV columns:
      Idx,predicted_group,real_group,real_interval
    """
    out_dir = os.path.normpath(os.path.join('..', 'DOGI-Gconf', 'DOGI-PLog'))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"Model{model_suffix}")

    idx_val_u64 = np.asarray(idx_val, dtype=np.uint64)
    order = np.argsort(idx_val_u64, kind='stable')

    out_idx = idx_val_u64[order]
    out_pred = np.asarray(pred_val, dtype=np.int64)[order]
    out_real = np.asarray(y_val, dtype=np.int64)[order]
    out_interval = np.asarray(real_interval_val, dtype=np.float64)[order]

    df_out = pd.DataFrame(
        {
            'idx': out_idx,
            'predicted_group': out_pred,
            'real_group': out_real,
            'real_interval': out_interval,
        }
    )
    df_out.to_csv(out_path, index=False)
    print(f"Saved validation predictions to: {out_path}")


def save_thresholds_for_bir(thresholds: np.ndarray, model_suffix: str):
    """Save thresholds for DOGI-BIR in the required text format.

    Format requirements:
      - Path: ../DOGI-Gconf/DOGI-BIR/ (relative to this script directory)
      - Filename: Model{model_suffix}
      - Lines: 0, then (thresholds/_PPS) as integers without decimals, then -1
        For 10 groups, this yields 11 lines total.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.normpath(os.path.join(script_dir, '..', 'DOGI-Gconf', 'DOGI-BIR'))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"Model{model_suffix}")

    thr = np.asarray(thresholds, dtype=np.float64)
    scaled = thr / float(_PPS)
    scaled_int = np.floor(scaled).astype(np.int64)

    if scaled_int.size != (NUM_GROUPS - 1):
        print(
            f"[WARN] thresholds count={scaled_int.size} but expected {NUM_GROUPS - 1} for NUM_GROUPS={NUM_GROUPS}. "
            f"BIR file will have {scaled_int.size + 2} lines (0 + thresholds + -1)."
        )

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("0\n")
        for v in scaled_int.tolist():
            f.write(f"{int(v)}\n")
        f.write("-1\n")

    print(f"Saved BIR thresholds to: {out_path}")


def save_hot_stub(model_suffix: str, hot_size: int = 3, hot_invalid: float = 0.95, p_hot: float = 0.80):
    """Save a HOT config stub (one line: size invalid_ratio hot_traffic)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.normpath(os.path.join(script_dir, '..', 'DOGI-Gconf', 'DOGI-HOT'))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"Model{model_suffix}")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"{hot_size} {hot_invalid} {p_hot}\n")
    print(f"Saved HOT stub to: {out_path}")


def load_and_prepare_data(csv_path, thresholds=DEFAULT_THRESHOLDS, train_ratio=0.9):
    """
    Load CSV data and prepare train/validation splits.
    
    Args:
        csv_path: Path to training CSV file
        thresholds: NumPy array of threshold boundaries
        train_ratio: Ratio of training data (0.9 = 90% train, 10% val)
    
    Returns:
        X_train, X_val, y_train, y_val, scaler, num_groups, thresholds_used, idx_train, idx_val, real_interval_train, real_interval_val
    """
    print(f"Loading data from: {csv_path}")
    
    # Load entire CSV into memory
    # load CSV file with maximum 400000 lines, randomly shuffle rows
    df = pd.read_csv(csv_path, nrows=400000)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle rows
    print(f"Total samples loaded: {len(df)}")

    # Basic stats for real_interval (raw + valid-only if sentinel -1 exists)
    real_interval = df[TARGET_FEATURE].to_numpy(dtype=np.float32, copy=False)

    raw_min = float(np.nanmin(real_interval))
    raw_max = float(np.nanmax(real_interval))
    print(f"real_interval min/max (raw): {raw_min:.4f} / {raw_max:.4f}")

    # Basic stats for specific feature values
    # Interval == 255
    interval_arr = df['Interval'].to_numpy(dtype=np.float32, copy=False)
    interval_non_nan = ~np.isnan(interval_arr)
    interval_total = int(np.sum(interval_non_nan))
    interval_255_count = int(np.sum(interval_non_nan & (interval_arr == 255)))
    interval_255_pct = (interval_255_count / interval_total * 100.0) if interval_total > 0 else 0.0
    print(f"Interval == 255: {interval_255_count}/{interval_total} ({interval_255_pct:.2f}%)")

    # FreqBits2 == 0/1
    freqbits2_arr = df['FreqBits2'].to_numpy(dtype=np.float32, copy=False)
    fb2_non_nan = ~np.isnan(freqbits2_arr)
    fb2_total = int(np.sum(fb2_non_nan))
    fb2_0_count = int(np.sum(fb2_non_nan & (freqbits2_arr == 0)))
    fb2_1_count = int(np.sum(fb2_non_nan & (freqbits2_arr == 1)))
    fb2_0_pct = (fb2_0_count / fb2_total * 100.0) if fb2_total > 0 else 0.0
    fb2_1_pct = (fb2_1_count / fb2_total * 100.0) if fb2_total > 0 else 0.0
    print(f"FreqBits2 == 0: {fb2_0_count}/{fb2_total} ({fb2_0_pct:.2f}%)")
    print(f"FreqBits2 == 1: {fb2_1_count}/{fb2_total} ({fb2_1_pct:.2f}%)")

    # SegFreq max
    segfreq_arr = df['SegFreq'].to_numpy(dtype=np.float32, copy=False)
    segfreq_max = float(np.nanmax(segfreq_arr))
    print(f"SegFreq max: {segfreq_max:.4f}")

    # Distribution of Interval/FreqBits/FreqBits2 among rows where RealInterval == 1
    ri1_mask = np.isfinite(real_interval) & (real_interval == np.float32(1.0))
    ri1_count = int(np.sum(ri1_mask))
    ri1_pct = (ri1_count / int(real_interval.size) * 100.0) if real_interval.size > 0 else 0.0
    print(f"RealInterval == 1: {ri1_count}/{int(real_interval.size)} ({ri1_pct:.2f}%)")
    if ri1_count > 0:
        for col in ['Interval', 'FreqBits', 'FreqBits2']:
            arr = df[col].to_numpy(dtype=np.float64, copy=False)
            sub = arr[ri1_mask]
            sub = sub[np.isfinite(sub)]
            if sub.size == 0:
                print(f"  {col}: no finite values")
                continue
            q25, q50, q75 = np.percentile(sub, [25, 50, 75])
            print(
                f"  {col}: n={sub.size}, mean={float(np.mean(sub)):.4f}, std={float(np.std(sub)):.4f}, "
                f"min={float(np.min(sub)):.4f}, p25={float(q25):.4f}, p50={float(q50):.4f}, p75={float(q75):.4f}, max={float(np.max(sub)):.4f}"
            )

    # Share of samples with large intervals
    large_thr = np.float32(LARGE_INTERV)
    non_nan = ~np.isnan(real_interval)
    large_mask = non_nan & (real_interval >= large_thr)
    large_count = int(np.sum(large_mask))
    total_count = int(real_interval.size)
    large_pct = (large_count / total_count * 100.0) if total_count > 0 else 0.0
    print(f"RealInterval >= {int(large_thr)}: {large_count}/{total_count} ({large_pct:.2f}%)")

    '''
    invalid_mask = (real_interval == -1)
    invalid_count = int(np.sum(invalid_mask))
    if invalid_count > 0:
        valid_mask = ~invalid_mask
        if np.any(valid_mask):
            valid_min = float(np.nanmin(real_interval[valid_mask]))
            valid_max = float(np.nanmax(real_interval[valid_mask]))
            print(
                f"real_interval min/max (valid only, excluding -1): {valid_min:.4f} / {valid_max:.4f} "
                f"(excluded {invalid_count} rows)"
            )

            valid_large_count = int(np.sum(large_mask & valid_mask))
            valid_total = int(np.sum(valid_mask))
            valid_large_pct = (valid_large_count / valid_total * 100.0) if valid_total > 0 else 0.0
            print(
                f"RealInterval >= {int(large_thr)} (valid only, excluding -1): "
                f"{valid_large_count}/{valid_total} ({valid_large_pct:.2f}%)"
            )
        else:
            print(f"Warning: all real_interval values are -1 (count={invalid_count}).")
    '''
    
    # If thresholds is None, compute balanced thresholds from loaded samples.
    thresholds_used = thresholds
    if thresholds is None:
        global DEFAULT_THRESHOLDS4
        DEFAULT_THRESHOLDS4 = compute_balanced_thresholds(real_interval, NUM_GROUPS).astype(np.float32)
        thresholds_used = DEFAULT_THRESHOLDS4
        print("\nComputed balanced thresholds -> DEFAULT_THRESHOLDS4: ")

    # Extract features and target (+ keep Idx/RealInterval for later use)
    # Idx is unsigned long long in C++ (uint64)
    idx_all = df['Idx'].to_numpy(copy=False)
    try:
        idx_all = idx_all.astype(np.uint64, copy=False)
    except Exception:
        idx_all = idx_all.astype(np.uint64)

    X = df[INPUT_FEATURES].values
    y = np.digitize(df[TARGET_FEATURE].values, thresholds_used)
    
    num_groups = len(thresholds_used) + 1
    print(f"Number of groups: {num_groups}")
    print(
        "Thresholds: "
        + _format_float_array(thresholds_used, precision=4)
    )
    print(
        "Thresholds (PPS): "
        + _format_float_array(thresholds_used / _PPS, precision=6)
    )

    # Group distribution (after digitize)
    print("\nGroup distribution by digitized RealInterval:")
    y_np = np.asarray(y, dtype=np.int64)
    total_y = int(y_np.size)
    counts = np.bincount(y_np, minlength=num_groups)
    for group_id in range(num_groups):
        pct = (counts[group_id] / total_y) * 100.0
        print(f"  Group {group_id}: {counts[group_id]}/{total_y} ({pct:.2f}%)")

    # If sentinel -1 exists, also show distribution excluding those rows
    '''
    if invalid_count > 0 and 'valid_mask' in locals() and np.any(valid_mask):
        y_valid = y_np[valid_mask]
        total_valid = int(y_valid.size)
        print("\nGroup distribution (valid only, excluding -1):")
        counts_valid = np.bincount(y_valid, minlength=num_groups)
        for group_id in range(num_groups):
            pct = (counts_valid[group_id] / total_valid) * 100.0 if total_valid > 0 else 0.0
            print(f"  Group {group_id}: {counts_valid[group_id]}/{total_valid} ({pct:.2f}%)")
    '''
    
    # Split into train (90%) and validation (10%)
    X_train, X_val, y_train, y_val, idx_train, idx_val, real_interval_train, real_interval_val = train_test_split(
        X, y, idx_all, real_interval,
        train_size=train_ratio, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    print(f"Scaler mean: {scaler.mean_}")
    print(f"Scaler scale: {scaler.scale_}")
    
    return (
        X_train,
        X_val,
        y_train,
        y_val,
        scaler,
        num_groups,
        thresholds_used,
        idx_train,
        idx_val,
        real_interval_train,
        real_interval_val,
    )


def evaluate_on_validation_set(model, X_val, y_val, device, num_groups: int):
    """Run inference on validation set and print overall + per-true-group accuracy."""
    print("\n" + "=" * 30)
    print("Final Validation Inference")
    print("=" * 30)

    model.eval()
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_np = np.asarray(y_val, dtype=np.int64)

    with torch.no_grad():
        outputs = model(X_val_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.int64)

    if y_val_np.size == 0:
        print("Validation set is empty; skipping evaluation.")
        return

    overall_acc = float(np.mean(preds == y_val_np) * 100.0)
    print(f"Overall accuracy: {overall_acc:.2f}% ({int(np.sum(preds == y_val_np))}/{y_val_np.size})")

    print("Per-group accuracy (true group n -> predicted n):")
    for group_id in range(num_groups):
        mask = (y_val_np == group_id)
        group_total = int(np.sum(mask))
        if group_total == 0:
            print(f"  Group {group_id}: N/A (0 samples)")
            continue
        group_correct = int(np.sum(preds[mask] == group_id))
        group_acc = (group_correct / group_total) * 100.0
        print(f"  Group {group_id}: {group_acc:.2f}% ({group_correct}/{group_total})")


# =============================================================================
# Training
# =============================================================================

def train_model(model, X_train, y_train, X_val, y_val, device,
                lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY,
                num_epochs=DEFAULT_NUM_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
                print_flag: int = 1):
    """
    Train the neural network model.
    
    Args:
        model: DynamicNN model instance
        X_train, y_train: Training data
        X_val, y_val: Validation data
        device: torch device (cuda or cpu)
        lr: Learning rate
        weight_decay: L2 regularization parameter
        num_epochs: Number of training epochs
        batch_size: Batch size for training
    """
    if print_flag != 0:
        print("\n" + "="*30)
        print("Starting Training")
        print("="*30)
        print(f"Device: {device}")
        print(f"Learning rate: {lr}")
        print(f"Weight decay: {weight_decay}")
        print(f"Batch size: {batch_size}")
        print(f"Number of epochs: {num_epochs}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop
    num_batches = (len(X_train) + batch_size - 1) // batch_size
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # Shuffle training data
        indices = torch.randperm(len(X_train_tensor))
        
        for i in range(0, len(X_train_tensor), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X = X_train_tensor[batch_indices]
            batch_y = y_train_tensor[batch_indices]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            epoch_loss += loss.item() * batch_y.size(0)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        # Epoch statistics
        avg_loss = epoch_loss / total
        train_acc = 100.0 * correct / total
        
        # Validation accuracy (no gradient computation)
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            _, val_predicted = torch.max(val_outputs, 1)
            val_correct = (val_predicted == y_val_tensor).sum().item()
            val_acc = 100.0 * val_correct / len(y_val)
            
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    if print_flag != 0:
        print("\nTraining completed!")


# =============================================================================
# Model Saving (for C++ inference using raw weights)
# =============================================================================

def save_model_for_cpp_inference(model, scaler, thresholds, output_dir):
    """
    Save model weights in format compatible with mlp_inference.cc.
    Exports raw weight matrices and biases as binary files.
    
    Model structure (matching mlp_inference.h):
      Layer 1: [6 x 32] + bias[32]  (W1, b1)
      Layer 2: [32 x 32] + bias[32] (W2, b2)
      Layer 3: [32 x 10] + bias[10] (W3, b3)
    
    Args:
        model: Trained DynamicNN model
        scaler: Fitted StandardScaler
        thresholds: NumPy array of group thresholds
        output_dir: Directory to save model files
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    model = model.cpu()
    
    print(f"\nSaving model weights to: {output_dir}")
    
    # Extract weights and biases from PyTorch model
    # Layer structure: Linear(6, 32), ReLU, Dropout, Linear(32, 32), ReLU, Dropout, Linear(32, 10)
    state_dict = model.state_dict()
    
    # Layer 1: model.model.0.weight and model.model.0.bias
    W1 = state_dict['model.0.weight'].numpy()  # Shape: [32, 6]
    b1 = state_dict['model.0.bias'].numpy()    # Shape: [32]
    
    # Layer 2: model.model.3.weight and model.model.3.bias
    W2 = state_dict['model.3.weight'].numpy()  # Shape: [32, 32]
    b2 = state_dict['model.3.bias'].numpy()    # Shape: [32]
    
    # Layer 3: model.model.6.weight and model.model.6.bias
    W3 = state_dict['model.6.weight'].numpy()  # Shape: [10, 32]
    b3 = state_dict['model.6.bias'].numpy()    # Shape: [10]
    
    # Save weights as raw binary files (float32, row-major)
    W1.astype(np.float32).tofile(os.path.join(output_dir, 'W1.bin'))
    b1.astype(np.float32).tofile(os.path.join(output_dir, 'b1.bin'))
    W2.astype(np.float32).tofile(os.path.join(output_dir, 'W2.bin'))
    b2.astype(np.float32).tofile(os.path.join(output_dir, 'b2.bin'))
    W3.astype(np.float32).tofile(os.path.join(output_dir, 'W3.bin'))
    b3.astype(np.float32).tofile(os.path.join(output_dir, 'b3.bin'))
    
    print(f"✓ Saved W1.bin: shape {W1.shape} = {W1.size} floats ({W1.nbytes} bytes)")
    print(f"✓ Saved b1.bin: shape {b1.shape} = {b1.size} floats ({b1.nbytes} bytes)")
    print(f"✓ Saved W2.bin: shape {W2.shape} = {W2.size} floats ({W2.nbytes} bytes)")
    print(f"✓ Saved b2.bin: shape {b2.shape} = {b2.size} floats ({b2.nbytes} bytes)")
    print(f"✓ Saved W3.bin: shape {W3.shape} = {W3.size} floats ({W3.nbytes} bytes)")
    print(f"✓ Saved b3.bin: shape {b3.shape} = {b3.size} floats ({b3.nbytes} bytes)")
    
    # Save scaler parameters (mean and scale)
    scaler_path = os.path.join(output_dir, 'scaler_params.npz')
    np.savez(scaler_path, mean=scaler.mean_.astype(np.float32), 
             scale=scaler.scale_.astype(np.float32))
    print(f"✓ Saved scaler_params.npz: mean and scale arrays")
    # Also save plain binaries for C++ loader (avoid npz parsing on device)
    mean_path = os.path.join(output_dir, 'mean.bin')
    scale_path = os.path.join(output_dir, 'scale.bin')
    scaler.mean_.astype(np.float32).tofile(mean_path)
    scaler.scale_.astype(np.float32).tofile(scale_path)
    print(f"✓ Saved mean.bin / scale.bin (float32, {scaler.mean_.size} values each)")
    
    # Save thresholds
    threshold_path = os.path.join(output_dir, 'thresholds.npy')
    np.save(threshold_path, thresholds)
    print(f"✓ Saved thresholds.npy")
    # Plain binary thresholds for C++ (optional)
    thresholds.astype(np.float32).tofile(os.path.join(output_dir, 'thresholds.bin'))
    
    # Also save PyTorch state dict for retraining
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))
    print(f"✓ Saved model.pth (PyTorch state dict for retraining)")

    print(f"\n{'='*70}")
    print(f"Model export complete!")
    print(f"{'='*70}")
    print(f"\nC++ integration instructions (mlp_inference.cc):")
    print(f"  1. Load binary weight files: W1.bin, W2.bin, W3.bin")
    print(f"  2. Load binary bias files: b1.bin, b2.bin, b3.bin")
    print(f"  3. Load scaler parameters from: scaler_params.npz")
    print(f"     - Apply: normalized_input = (input - mean) / scale")
    print(f"  4. Matrix dimensions:")
    print(f"     - W1: [{W1.shape[0]} x {W1.shape[1]}] (hidden1 x input)")
    print(f"     - W2: [{W2.shape[0]} x {W2.shape[1]}] (hidden2 x hidden1)")
    print(f"     - W3: [{W3.shape[0]} x {W3.shape[1]}] (output x hidden2)")
    print(f"  5. Forward pass: X @ W1^T + b1 -> ReLU -> @ W2^T + b2 -> ReLU -> @ W3^T + b3")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python model_trainer.py <training_csv> [print_flag]")
        print("\nExample:")
        print("  python model_trainer.py 0_training.csv 1")
        print("\nprint_flag:")
        print("  0: minimal prints (start, per-epoch acc, final completed/export)")
        print("  1: verbose prints (default)")
        print("\nOutput files will be saved to the same directory:")
        print("  - W1.bin, W2.bin, W3.bin (weight matrices)")
        print("  - b1.bin, b2.bin, b3.bin (bias vectors)")
        print("  - scaler_params.npz (StandardScaler parameters)")
        print("  - thresholds.npy (group thresholds)")
        print("  - model.pth (PyTorch state dict for retraining)")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    print_flag = 0
    if len(sys.argv) >= 3:
        try:
            print_flag = int(sys.argv[2])
        except Exception:
            print_flag = 0
    
    # Validate input
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    # Output directory: <csv_dir>/TrainedModel/Model{N}/
    csv_dir = os.path.dirname(csv_path)
    if not csv_dir:
        csv_dir = '.'

    base_name = os.path.basename(csv_path)
    # Accept both numeric and non-numeric prefixes.
    # Examples:
    #   0_training.csv  -> Model0
    #   t1_training.csv -> Modelt1
    m = re.match(r'^(?P<prefix>.+)_training\.csv$', base_name)
    if m:
        model_suffix = m.group('prefix')
    else:
        # Fallback if filename doesn't follow *_training.csv
        model_suffix = os.path.splitext(base_name)[0]

    output_dir = os.path.join(csv_dir, 'TrainedModel', f'Model{model_suffix}')
    os.makedirs(output_dir, exist_ok=True)
    
    if print_flag == 0:
        print("Model training start")

    suppress = (print_flag == 0)

    with _maybe_suppress_stdout(suppress):
        print("="*70)
        print("DOGI Model Trainer - CPU Single-threaded")
        print("="*70)
        print(f"CSV file: {csv_path}")
        print(f"Output directory: {output_dir}")
        print(f"Model architecture: {INPUT_DIM} -> {HIDDEN_LAYERS[0]} -> {HIDDEN_LAYERS[1]} -> {NUM_GROUPS}")
        print(f"Hyperparameters:")
        print(f"  - Learning rate: {DEFAULT_LR}")
        print(f"  - Weight decay: {DEFAULT_WEIGHT_DECAY}")
        print(f"  - Dropout rate: {DEFAULT_DROPOUT_RATE}")
        print(f"  - Batch size: {DEFAULT_BATCH_SIZE}")
        print(f"  - Epochs: {DEFAULT_NUM_EPOCHS}")
        print("="*70)
    
    # Force CPU device
    device = torch.device('cpu')

    with _maybe_suppress_stdout(suppress):
        print(f"\nDevice: {device} (single-threaded)")
        
        # Load and prepare data
        X_train, X_val, y_train, y_val, scaler, num_groups, thresholds_used, idx_train, idx_val, real_interval_train, real_interval_val = load_and_prepare_data(
            csv_path, thresholds=None, train_ratio=0.9
        )
        
        # Create model
        model = DynamicNN(
            input_size=INPUT_DIM,
            num_classes=num_groups,
            hidden_layers=HIDDEN_LAYERS,
            dropout_rate=DEFAULT_DROPOUT_RATE
        ).to(device)
        
        print(f"\nModel architecture:")
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    train_model(
        model, X_train, y_train, X_val, y_val, device,
        lr=DEFAULT_LR,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        num_epochs=DEFAULT_NUM_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
        print_flag=print_flag,
    )

    with _maybe_suppress_stdout(suppress):
        # Final evaluation on the held-out validation set
        evaluate_on_validation_set(model, X_val, y_val, device, num_groups)

        # Save per-sample validation predictions sorted by Idx
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.FloatTensor(X_val).to(device))
            val_pred = torch.argmax(val_outputs, dim=1).cpu().numpy()
        save_validation_predictions(output_dir, model_suffix, idx_val, y_val, real_interval_val, val_pred)
        
        # Save thresholds in DOGI-BIR format (text, PPS-scaled, no decimals)
        save_thresholds_for_bir(thresholds_used, model_suffix)
        # Save HOT stub alongside PLog/BIR outputs for the optimizer.
        save_hot_stub(model_suffix)
        
        # Save model for C++ inference
        save_model_for_cpp_inference(model, scaler, thresholds_used, output_dir)

        # Mark this model as the latest for inference (text pointer inside TrainedModel/latest).
        latest_path = os.path.join(os.path.dirname(output_dir), "latest")
        try:
            with open(latest_path, "w") as f:
                f.write(os.path.basename(output_dir).strip() + "\n")
            print(f"Updated latest model pointer: {latest_path} -> {os.path.basename(output_dir)}")
        except Exception as e:
            print(f"Warning: failed to update latest pointer at {latest_path}: {e}")

        print("\n" + "="*70)
        print("Training pipeline completed successfully!")
        print("="*70)

    if print_flag == 0:
        print("Model training completed, and export completed")


if __name__ == "__main__":
    main()
