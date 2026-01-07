import os
import math
import json
import csv
import shutil
import warnings
from datetime import datetime
from typing import List, Optional
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.manifold import TSNE
import mne
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


DATA_ROOT = "./data"        
EPOCHS_SUFFIX = "-epo.fif"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

TARGET_SFREQ = 250
TMIN, TMAX = -0.2, 0.8         
BANDPASS = (1.0, 40.0)
NOTCH = 50.0

BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3

DANN_WARMUP_EPOCHS = 5
LAMBDA_DOMAIN = 0.5

FIXED_CHANNELS: Optional[List[str]] = None
ARTIFACT_DIR = "./artifacts"


TSNE_MAX_BATCHES = 12        
TSNE_PERPLEXITY = 30


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def list_subject_epoch_files(root, suffix):
    files = {}
    for fn in sorted(os.listdir(root)):
        if fn.endswith(suffix):
            sid = fn.replace(suffix, "")
            files[sid] = os.path.join(root, fn)
    if not files:
        raise RuntimeError("No epoch files found")
    return files

def safe_common_channels(epochs_list, fixed=None):
    if fixed:
        return [ch for ch in fixed if all(ch in e.ch_names for e in epochs_list)]
    common = set(epochs_list[0].ch_names)
    for e in epochs_list[1:]:
        common &= set(e.ch_names)
    return sorted(list(common))

def preprocess_epochs(ep, chs):
    ep = ep.copy().pick(chs)

    
    try:
        ep = ep.notch_filter(NOTCH, verbose=False)
    except Exception:
        pass

    try:
        ep = ep.filter(BANDPASS[0], BANDPASS[1], verbose=False)
    except Exception:
        pass

    # resample
    if int(ep.info["sfreq"]) != TARGET_SFREQ:
        ep = ep.resample(TARGET_SFREQ)

    # crop time window (train-time)
    ep = ep.crop(TMIN, TMAX)

    # baseline only if we have pre-0 samples
    if ep.tmin < 0:
        try:
            ep = ep.apply_baseline((None, 0.0))
        except Exception:
            pass

    return ep

def epochs_to_xy(ep):
    X = ep.get_data().astype(np.float32)   # (N,C,T)
    y = ep.events[:, 2].astype(np.int64)
    return X, y

def fit_scaler(X):
    N, C, T = X.shape
    flat = X.transpose(0, 2, 1).reshape(-1, C)
    sc = StandardScaler().fit(flat)
    return sc

def apply_scaler(X, sc):
    N, C, T = X.shape
    flat = X.transpose(0, 2, 1).reshape(-1, C)
    return sc.transform(flat).reshape(N, T, C).transpose(0, 2, 1).astype(np.float32)

def save_metrics_csv(path, rows, header=("subject", "bal_acc", "kappa")):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for r in rows:
            w.writerow(r)

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def save_cm(y_true, y_pred, class_names, out_png, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

@torch.no_grad()
def collect_features_and_labels(model, loader, max_batches=10):
    """Collect EEGNet features (pre-head) + class/domain labels from loader."""
    model.eval()
    Z, Y, D = [], [], []
    for bi, (x, y, d) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x = x.to(DEVICE)
        z = model.fe(x).cpu().numpy()  # (B,F)
        Z.append(z)
        Y.append(y.numpy())
        D.append(d.numpy())
    Z = np.concatenate(Z, axis=0)
    Y = np.concatenate(Y, axis=0)
    D = np.concatenate(D, axis=0)
    return Z, Y, D

def save_tsne(Z, labels, out_png, title, perplexity=30):
    if Z.shape[0] < 10:
        return
    # t-SNE perplexity must be < n_samples
    perp = min(perplexity, max(5, (Z.shape[0] - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
        random_state=SEED
    )
    emb = tsne.fit_transform(Z)

    plt.figure()
    uniq = np.unique(labels)
    for lab in uniq:
        m = labels == lab
        plt.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.7, label=str(lab))
    plt.title(title)
    plt.legend(markerscale=2, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

class TensorDataset3(Dataset):
    def __init__(self, X, y, d):
        self.X, self.y, self.d = X, y, d
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i], self.d[i]


class GRLFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l):
        ctx.l = l
        return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.l * g, None

class GRL(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = 0.0
    def forward(self, x):
        return GRLFn.apply(x, self.l)

class EEGNetFE(nn.Module):
    def __init__(self, C, T):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, (1, 64), bias=False)
        self.conv2 = nn.Conv2d(8, 16, (C, 1), groups=8, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.pool = nn.AvgPool2d((1, 4))
        self.drop = nn.Dropout(0.25)
        self.sep = nn.Conv2d(16, 16, (1, 16), bias=False)
        self.pool2 = nn.AvgPool2d((1, 8))

        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)
            z = self._forward_conv(dummy)
            self.out_dim = z.shape[1]

    def _forward_conv(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = F.elu(x)
        x = self.pool(x)
        x = self.drop(x)
        x = self.sep(x)
        x = self.pool2(x)
        return x.flatten(1)

    def forward(self, x):
        x = x.unsqueeze(1)   # (B,1,C,T)
        return self._forward_conv(x)

class DANN_EEGNet(nn.Module):
    def __init__(self, C, T, n_classes, n_domains):
        super().__init__()
        self.fe = EEGNetFE(C, T)
        self.grl = GRL()
        self.label_head = nn.Linear(self.fe.out_dim, n_classes)
        self.domain_head = nn.Linear(self.fe.out_dim, n_domains)

    def forward(self, x, alpha: float):
        z = self.fe(x)
        y_logits = self.label_head(z)
        self.grl.l = float(alpha)
        d_logits = self.domain_head(self.grl(z))
        return y_logits, d_logits

    @torch.no_grad()
    def forward_label(self, x):
        z = self.fe(x)
        return self.label_head(z)


def dann_alpha(ep, total):
    if ep < DANN_WARMUP_EPOCHS:
        return 0.0
    p = (ep - DANN_WARMUP_EPOCHS + 1) / max(1, (total - DANN_WARMUP_EPOCHS))
    return float(2 / (1 + math.exp(-10 * p)) - 1)

@torch.no_grad()
def eval_model(model, loader):
    model.eval()
    P, Y = [], []
    for x, y, _ in loader:
        x = x.to(DEVICE)
        out = model.forward_label(x)
        P.append(out.argmax(1).cpu().numpy())
        Y.append(y.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    return balanced_accuracy_score(Y, P), cohen_kappa_score(Y, P), Y, P

def run_loso():
    set_seed(SEED)
    ensure_dir(ARTIFACT_DIR)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(ARTIFACT_DIR, run_tag)
    ensure_dir(run_dir)

    metrics_csv = os.path.join(run_dir, f"loso_metrics_{run_tag}.csv")


    files = list_subject_epoch_files(DATA_ROOT, EPOCHS_SUFFIX)
    subjects = list(files.keys())

    epochs_all = {s: mne.read_epochs(f, preload=True, verbose=False) for s, f in files.items()}
    common_chs = safe_common_channels(list(epochs_all.values()), FIXED_CHANNELS)
    xy = {s: epochs_to_xy(preprocess_epochs(ep, common_chs)) for s, ep in epochs_all.items()}

    bals, kaps = [], []
    all_rows = []

    all_targets_true = []
    all_targets_pred = []
    class_names_last = None

    for tgt in subjects:
        srcs = [s for s in subjects if s != tgt]

        Xs = np.concatenate([xy[s][0] for s in srcs])
        ys_raw = np.concatenate([xy[s][1] for s in srcs])
        Xt, yt_raw = xy[tgt]

        common = np.intersect1d(np.unique(ys_raw), np.unique(yt_raw))
        if len(common) < 2:
            print(f"{tgt}: skip (common classes < 2)")
            continue

        ms = np.isin(ys_raw, common)
        mt = np.isin(yt_raw, common)

        Xs, ys_raw = Xs[ms], ys_raw[ms]
        Xt, yt_raw = Xt[mt], yt_raw[mt]

        label_map = {int(c): i for i, c in enumerate(common)}
        ys = np.array([label_map[int(v)] for v in ys_raw], dtype=np.int64)
        yt = np.array([label_map[int(v)] for v in yt_raw], dtype=np.int64)

        n_classes = len(common)

        class_names = [str(int(c)) for c in common]
        class_names_last = class_names

        dom_map = {s: i for i, s in enumerate(srcs)}
        dom_map[tgt] = len(srcs)

        ds_full = np.concatenate([np.full(len(xy[s][0]), dom_map[s], dtype=np.int64) for s in srcs])
        ds = ds_full[ms]
        dt = np.full(len(yt), dom_map[tgt], dtype=np.int64)

        sc = fit_scaler(Xs)
        Xs, Xt = apply_scaler(Xs, sc), apply_scaler(Xt, sc)

        src_loader = DataLoader(
            TensorDataset3(torch.tensor(Xs), torch.tensor(ys), torch.tensor(ds)),
            BATCH_SIZE, shuffle=True, drop_last=True
        )

        tgt_loader = DataLoader(
            TensorDataset3(torch.tensor(Xt), torch.tensor(yt), torch.tensor(dt)),
            BATCH_SIZE, shuffle=True, drop_last=True
        )

        eval_loader = DataLoader(
            TensorDataset3(torch.tensor(Xt), torch.tensor(yt), torch.tensor(dt)),
            BATCH_SIZE, shuffle=False
        )

        model = DANN_EEGNet(Xs.shape[1], Xs.shape[2], n_classes, len(srcs) + 1).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=LR)

        tgt_iter = itertools.cycle(tgt_loader)

        for ep in range(EPOCHS):
            a = dann_alpha(ep, EPOCHS)
            model.train()

            for (xs, ys_b, ds_b) in src_loader:
                xt, _, dt_b = next(tgt_iter)

                xs, ys_b, ds_b = xs.to(DEVICE), ys_b.to(DEVICE), ds_b.to(DEVICE)
                xt, dt_b = xt.to(DEVICE), dt_b.to(DEVICE)

                opt.zero_grad()

                y_s, d_s = model(xs, a)
                _,   d_t = model(xt, a)

                loss = F.cross_entropy(y_s, ys_b) + LAMBDA_DOMAIN * (
                    F.cross_entropy(d_s, ds_b) + F.cross_entropy(d_t, dt_b)
                )

                loss.backward()
                opt.step()

        bal, kap, Y_true, Y_pred = eval_model(model, eval_loader)
        print(f"{tgt}: BalAcc={bal:.3f} Kappa={kap:.3f}")

        bals.append(bal)
        kaps.append(kap)

        all_rows.append((tgt, float(bal), float(kap)))
        save_metrics_csv(metrics_csv, [(tgt, float(bal), float(kap))])

        cm_path = os.path.join(ARTIFACT_DIR, f"cm_{run_tag}_{tgt}.png")
        save_cm(Y_true, Y_pred, class_names, cm_path, title=f"{tgt} Confusion Matrix")

        Zt, Yt, Dt = collect_features_and_labels(model, eval_loader, max_batches=TSNE_MAX_BATCHES)
        save_tsne(
            Zt, Yt,
            os.path.join(ARTIFACT_DIR, f"tsne_{run_tag}_{tgt}_class.png"),
            title=f"{tgt} t-SNE (features) by CLASS",
            perplexity=TSNE_PERPLEXITY
        )
        save_tsne(
            Zt, Dt,
            os.path.join(ARTIFACT_DIR, f"tsne_{run_tag}_{tgt}_domain.png"),
            title=f"{tgt} t-SNE (features) by DOMAIN",
            perplexity=TSNE_PERPLEXITY
        )

        all_targets_true.append(Y_true)
        all_targets_pred.append(Y_pred)

    print("\nFINAL:")
    mean_bal = float(np.mean(bals)) if bals else None
    mean_kap = float(np.mean(kaps)) if kaps else None
    print("BalAcc:", mean_bal)
    print("Kappa:", mean_kap)

    summary = {
        "run_tag": run_tag,
        "mean_bal_acc": mean_bal,
        "mean_kappa": mean_kap,
        "per_subject": [{"subject": r[0], "bal_acc": r[1], "kappa": r[2]} for r in all_rows],
        "config": {
            "DATA_ROOT": DATA_ROOT,
            "EPOCHS_SUFFIX": EPOCHS_SUFFIX,
            "TARGET_SFREQ": TARGET_SFREQ,
            "TMIN": TMIN,
            "TMAX": TMAX,
            "BANDPASS": BANDPASS,
            "NOTCH": NOTCH,
            "BATCH_SIZE": BATCH_SIZE,
            "EPOCHS": EPOCHS,
            "LR": LR,
            "DANN_WARMUP_EPOCHS": DANN_WARMUP_EPOCHS,
            "LAMBDA_DOMAIN": LAMBDA_DOMAIN,
            "TSNE_MAX_BATCHES": TSNE_MAX_BATCHES,
            "TSNE_PERPLEXITY": TSNE_PERPLEXITY,
        }
    }

    summary_path = os.path.join(run_dir, f"loso_summary_{run_tag}.json")
    save_json(summary_path, summary)
    print("Saved metrics:", metrics_csv)
    print("Saved summary:", summary_path)

    if all_targets_true and all_targets_pred and class_names_last is not None:
        Y_all = np.concatenate(all_targets_true)
        P_all = np.concatenate(all_targets_pred)
        cm_all_path = os.path.join(run_dir, f"cm_{run_tag}_ALL.png")
        save_cm(Y_all, P_all, class_names_last, cm_all_path, title="ALL Targets Confusion Matrix")
        print("Saved overall CM:", cm_all_path)

    zip_path = shutil.make_archive(
        os.path.join(ARTIFACT_DIR, f"artifacts_{run_tag}"),
        "zip",
        run_dir
    )
    print("Zipped artifacts:", zip_path)


if __name__ == "__main__":
    run_loso()
 вот код
```
