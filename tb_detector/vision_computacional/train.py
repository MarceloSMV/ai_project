"""
train.py — Entrenamiento del Clasificador de Tuberculosis (EfficientNet-B3 / ResNet-50)

ESTRATEGIA:
  1. Fine-tuning en dos fases:
     - Fase 1 (warm-up): Solo la cabeza lineal, backbone congelado.
     - Fase 2 (fine-tune): Todo el modelo descongelado con LR diferenciada.
  2. Loss: CrossEntropyLoss con pesos de clase para balancear TB.
  3. Scheduler: CosineAnnealingLR con warm-up manual.
  4. Métricas: Accuracy, F1-score por clase, mejor modelo en val_acc.
  5. Guardado atómico + hitos inmutables + backup por épocas.

Uso rápido:
    python train.py                          # EfficientNet-B3 (recomendado)
    python train.py --backbone resnet50      # ResNet-50
    python train.py --resume                 # Reanudar desde last.pt
    python train.py --reset                  # Empezar desde cero

SISTEMA DE PROTECCIÓN:
  - Guardado atómico (.tmp → .pt): sin corrupción por Ctrl+C.
  - best_acc se persiste en el checkpoint.
  - Hitos inmutables: milestone_70acc.pt ... 90acc.pt
"""

from __future__ import annotations
import os
import sys
import time
import shutil
import argparse
import warnings
from pathlib import Path
from datetime import datetime

_VC_DIR = os.path.dirname(os.path.abspath(__file__))
_TB_DIR = os.path.dirname(_VC_DIR)
if _VC_DIR not in sys.path:
    sys.path.insert(0, _VC_DIR)

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataloaders, CLASS_NAMES, NUM_CLASSES
from models.classifier import build_model

warnings.filterwarnings("ignore")


# ── Configuración por defecto ─────────────────────────────────────────────────
DEFAULT_CFG = {
    "root_dir":           _TB_DIR,
    "output_dir":         os.path.join(_VC_DIR, "runs"),
    "backbone":           "efficientnet",   # 'efficientnet' | 'resnet50'
    "img_size":           224,
    "batch_size":         32,
    # Fase 1 (cabeza congelada)
    "warmup_epochs":      5,
    "warmup_lr":          1e-3,
    # Fase 2 (fine-tune completo)
    "finetune_epochs":    30,
    "finetune_lr_head":   5e-4,
    "finetune_lr_backbone": 5e-5,
    "weight_decay":       1e-4,
    "dropout":            0.40,
    "num_workers":        4,
    "seed":               42,
    "keep_epoch_backups": 3,
    "epoch_backup_every": 5,
}


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_save(obj: dict, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    shutil.move(str(tmp), str(path))


def cleanup_epoch_backups(ckpt_dir: Path, keep: int):
    for old in sorted(ckpt_dir.glob("epoch_???.pt"))[:-keep]:
        try:
            old.unlink()
        except OSError:
            pass


def load_checkpoint(path: Path | None):
    if not path or not path.exists():
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "epoch" not in ckpt or "model" not in ckpt:
            print(f"  [WARN] {path.name}: checkpoint incompleto, ignorando.")
            return None
        return ckpt
    except Exception as e:
        print(f"  [WARN] No se pudo cargar {path.name}: {e}")
        return None


def compute_metrics(preds: list[int], labels: list[int]) -> dict:
    """Accuracy global + F1 por clase (macro)."""
    preds_np  = np.array(preds)
    labels_np = np.array(labels)
    acc       = (preds_np == labels_np).mean()

    f1_per_class = []
    for cls in range(NUM_CLASSES):
        tp = ((preds_np == cls) & (labels_np == cls)).sum()
        fp = ((preds_np == cls) & (labels_np != cls)).sum()
        fn = ((preds_np != cls) & (labels_np == cls)).sum()
        prec   = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1     = 2 * prec * recall / (prec + recall + 1e-8)
        f1_per_class.append(float(f1))

    return {"acc": float(acc), "f1_macro": float(np.mean(f1_per_class)),
            "f1_per_class": f1_per_class}


def freeze_backbone(model: nn.Module, backbone_name: str):
    """Congela todo excepto la cabeza de clasificación."""
    if backbone_name == "efficientnet":
        for p in model.features.parameters():
            p.requires_grad = False
    else:  # resnet50
        for name, p in model.named_parameters():
            if "fc" not in name:
                p.requires_grad = False


def unfreeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True


def get_optimizer_phase1(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """Solo parámetros de la cabeza."""
    head_params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(head_params, lr=cfg["warmup_lr"], weight_decay=cfg["weight_decay"])


def get_optimizer_phase2(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """LR diferenciada: mayor para la cabeza, menor para el backbone."""
    backbone_name = getattr(model, "_backbone_name", "efficientnet")
    if backbone_name == "efficientnet":
        head_params     = list(model.classifier.parameters())
        backbone_params = list(model.features.parameters())
    else:
        head_params     = list(model.fc.parameters())
        backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]

    return AdamW([
        {"params": head_params,     "lr": cfg["finetune_lr_head"]},
        {"params": backbone_params, "lr": cfg["finetune_lr_backbone"]},
    ], weight_decay=cfg["weight_decay"])


def train_epoch(model, loader, optimizer, criterion, device, epoch_num) -> tuple[float, list, list]:
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

        if (batch_idx + 1) % 20 == 0:
            print(f"  Epoch {epoch_num} | Batch {batch_idx+1}/{len(loader)} | Loss {loss.item():.4f}")

    return total_loss / len(loader), all_preds, all_labels


@torch.no_grad()
def val_epoch(model, loader, criterion, device) -> tuple[float, list, list]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / len(loader), all_preds, all_labels


def train(cfg: dict):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = cfg.get("backbone", "efficientnet")

    print(f"\n{'='*60}")
    print(f"  TBDetector — Clasificador {backbone.upper()}")
    print(f"  Dispositivo : {device}")
    print(f"  root_dir    : {cfg['root_dir']}")
    print(f"  output_dir  : {cfg['output_dir']}")
    print(f"{'='*60}\n")

    output_dir = Path(cfg["output_dir"])
    ckpt_dir   = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path   = output_dir / "training_log.csv"

    train_loader, val_loader, _ = get_dataloaders(
        root_dir    = cfg["root_dir"],
        batch_size  = cfg["batch_size"],
        img_size    = cfg["img_size"],
        num_workers = cfg["num_workers"],
        seed        = cfg["seed"],
    )

    # ── Pesos de clase para CrossEntropyLoss ─────────────────────────────────
    # health=3800, sick=3800, tb=1500 → más peso a TB
    counts        = np.array([3800.0, 3800.0, 1500.0])
    class_weights = torch.tensor(1.0 / counts * counts.sum(), dtype=torch.float32).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    model = build_model(
        num_classes=NUM_CLASSES,
        backbone=backbone,
        dropout=cfg["dropout"],
        pretrained=True,
    ).to(device)

    best_acc    = 0.0
    start_epoch = 1
    total_epochs = cfg["warmup_epochs"] + cfg["finetune_epochs"]
    phase = 1   # 1=warm-up cabeza, 2=fine-tune completo

    # ── Reanudar ─────────────────────────────────────────────────────────────
    resume_path  = None if cfg.get("reset") else ckpt_dir / "last.pt"
    ckpt_loaded  = load_checkpoint(resume_path)

    if ckpt_loaded:
        model.load_state_dict(ckpt_loaded["model"])
        start_epoch = ckpt_loaded["epoch"] + 1
        best_acc    = ckpt_loaded.get("best_acc", 0.0)
        phase       = ckpt_loaded.get("phase", 1)
        print(f"[Train] Reanudando desde época {start_epoch}  |  Mejor acc: {best_acc*100:.1f}%  |  Fase: {phase}")
        with open(log_path, "a") as f:
            if not log_path.exists():
                f.write("epoch,phase,train_loss,val_loss,val_acc,val_f1,best_acc,lr_head,time_s\n")
    else:
        print("[Train] Iniciando entrenamiento desde cero\n")
        with open(log_path, "w") as f:
            f.write("epoch,phase,train_loss,val_loss,val_acc,val_f1,best_acc,lr_head,time_s\n")

    epoch = start_epoch
    try:
        for epoch in range(start_epoch, total_epochs + 1):
            t0 = time.time()

            # ── Determinar fase ──────────────────────────────────────────────
            if epoch <= cfg["warmup_epochs"]:
                if phase != 1:
                    phase = 1
                    freeze_backbone(model, backbone)
                    print(f"\n  [FASE 1] Warm-up cabeza ({cfg['warmup_epochs']} épocas) ────────")
                if epoch == start_epoch or phase == 1:
                    optimizer  = get_optimizer_phase1(model, cfg)
                    scheduler  = CosineAnnealingLR(optimizer,
                                                   T_max=cfg["warmup_epochs"], eta_min=1e-5)
            else:
                if phase == 1:
                    phase = 2
                    unfreeze_all(model)
                    optimizer = get_optimizer_phase2(model, cfg)
                    scheduler = CosineAnnealingLR(optimizer,
                                                  T_max=cfg["finetune_epochs"], eta_min=1e-6)
                    print(f"\n  [FASE 2] Fine-tune completo ({cfg['finetune_epochs']} épocas) ──")

            # ── Entrenamiento ────────────────────────────────────────────────
            train_loss, train_preds, train_labels = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch
            )
            scheduler.step()

            # ── Validación ───────────────────────────────────────────────────
            val_loss, val_preds, val_labels = val_epoch(model, val_loader, criterion, device)
            metrics = compute_metrics(val_preds, val_labels)
            val_acc = metrics["acc"]
            val_f1  = metrics["f1_macro"]
            elapsed = time.time() - t0
            lr_head = optimizer.param_groups[0]["lr"]

            # F1 por clase
            f1_str = "  ".join(
                f"{CLASS_NAMES[i]}={metrics['f1_per_class'][i]*100:.1f}%"
                for i in range(NUM_CLASSES)
            )
            print(
                f"[Epoch {epoch:03d}/{total_epochs}] P{phase} | "
                f"Loss={train_loss:.4f} → {val_loss:.4f}  |  "
                f"Acc={val_acc*100:.1f}%  F1={val_f1*100:.1f}%  |  "
                f"Mejor={best_acc*100:.1f}%  |  LR={lr_head:.2e}  t={elapsed:.1f}s"
            )
            print(f"   F1 por clase: {f1_str}")

            # ── Log CSV ──────────────────────────────────────────────────────
            with open(log_path, "a") as f:
                f.write(f"{epoch},{phase},{train_loss:.6f},{val_loss:.6f},"
                        f"{val_acc:.6f},{val_f1:.6f},{best_acc:.6f},"
                        f"{lr_head:.8f},{elapsed:.1f}\n")

            # ── Checkpoint base ───────────────────────────────────────────────
            ckpt = {
                "epoch":     epoch,
                "phase":     phase,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_acc":   val_acc,
                "val_f1":    val_f1,
                "best_acc":  best_acc,
                "cfg":       cfg,
                "timestamp": datetime.now().isoformat(),
            }
            atomic_save(ckpt, ckpt_dir / "last.pt")

            # ── Mejor modelo ─────────────────────────────────────────────────
            if val_acc > best_acc:
                best_acc    = val_acc
                ckpt["best_acc"] = best_acc
                atomic_save(ckpt, ckpt_dir / "best.pt")
                print(f"  ★ NUEVO MEJOR: Acc={best_acc*100:.2f}%  F1={val_f1*100:.2f}%")

            # ── Hitos inmutables ──────────────────────────────────────────────
            for milestone in [0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95]:
                mfile = ckpt_dir / f"milestone_{int(milestone*100):02d}acc.pt"
                if val_acc >= milestone and not mfile.exists():
                    atomic_save(ckpt, mfile)
                    print(f"  [HITO] {mfile.name} — Acc={val_acc*100:.2f}%")

            # ── Backup por épocas ────────────────────────────────────────────
            if epoch % cfg.get("epoch_backup_every", 5) == 0:
                atomic_save(ckpt, ckpt_dir / f"epoch_{epoch:03d}.pt")
                cleanup_epoch_backups(ckpt_dir, cfg.get("keep_epoch_backups", 3))

    except KeyboardInterrupt:
        print(f"\n[Train] Pausado en época {epoch}. Mejor Acc: {best_acc*100:.2f}%")
        print("[Train] Para reanudar: python train.py --resume")
        return

    print(f"\n{'='*60}")
    print(f"  Entrenamiento completado.")
    print(f"  Mejor Accuracy Validación : {best_acc*100:.2f}%")
    print(f"  Checkpoint guardado en    : {ckpt_dir / 'best.pt'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TBDetector — Clasificador EfficientNet/ResNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root_dir",            default=DEFAULT_CFG["root_dir"])
    parser.add_argument("--output_dir",          default=DEFAULT_CFG["output_dir"])
    parser.add_argument("--backbone",            default=DEFAULT_CFG["backbone"],
                        choices=["efficientnet", "resnet50"])
    parser.add_argument("--img_size",     type=int,   default=DEFAULT_CFG["img_size"])
    parser.add_argument("--batch_size",   type=int,   default=DEFAULT_CFG["batch_size"])
    parser.add_argument("--warmup_epochs", type=int,  default=DEFAULT_CFG["warmup_epochs"])
    parser.add_argument("--finetune_epochs", type=int, default=DEFAULT_CFG["finetune_epochs"])
    parser.add_argument("--warmup_lr",     type=float, default=DEFAULT_CFG["warmup_lr"])
    parser.add_argument("--finetune_lr_head", type=float, default=DEFAULT_CFG["finetune_lr_head"])
    parser.add_argument("--finetune_lr_backbone", type=float, default=DEFAULT_CFG["finetune_lr_backbone"])
    parser.add_argument("--dropout",       type=float, default=DEFAULT_CFG["dropout"])
    parser.add_argument("--num_workers",   type=int,   default=DEFAULT_CFG["num_workers"])
    parser.add_argument("--resume",        action="store_true", help="Reanudar desde last.pt")
    parser.add_argument("--reset",         action="store_true", help="Empezar desde cero")

    args = parser.parse_args()
    cfg  = {**DEFAULT_CFG, **vars(args)}

    if cfg.get("reset"):
        last = Path(cfg["output_dir"]) / "checkpoints" / "last.pt"
        if last.exists():
            last.unlink()
            print("[Train] last.pt eliminado. Iniciando desde cero.")

    train(cfg)
