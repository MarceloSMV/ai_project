"""
dataset.py — TBDetector Dataset & DataLoader (Clasificador Puro)

Carga imágenes de radiografías con etiquetas a nivel de imagen:
  - health (0): Pulmones sanos
  - sick   (1): Enfermedad pulmonar (no TB)
  - tb     (2): Tuberculosis

Sin bounding boxes ni pseudo-anotaciones.
Las cajas delimitadoras se generan dinámicamente en inferencia via Grad-CAM.

Estructura esperada en root_dir:
  root_dir/
    imgs/
      health/   → radiografías sanas
      sick/     → radiografías enfermas (no TB)
      tb/       → radiografías con tuberculosis
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
import torchvision.transforms as T

# ── Etiquetas ─────────────────────────────────────────────────────────────────
CLASS_MAP   = {"health": 0, "sick": 1, "tb": 2}
CLASS_NAMES = ["health", "sick", "tb"]
NUM_CLASSES = 3


# ── Transforms ────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(split: str, img_size: int = 224):
    """
    Pipelines de augmentation diferenciados:
      - 'train': augmentation moderado + normalización ImageNet
      - 'val'/'test': solo resize + normalización ImageNet
    """
    if split == "train":
        return T.Compose([
            T.Resize((img_size + 32, img_size + 32)),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=12),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.04),
            T.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.9, 1.10)),
            T.RandomPerspective(distortion_scale=0.15, p=0.25),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            T.RandomErasing(p=0.15, scale=(0.02, 0.08)),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ── Dataset ───────────────────────────────────────────────────────────────────
class TBDataset(Dataset):
    """
    Dataset de clasificación de radiografías para TB.
    Retorna (imagen_tensor, label_int) para entrenamiento con CrossEntropyLoss.
    """

    def __init__(
        self,
        root_dir:  str,
        split:     str = "train",
        img_size:  int = 224,
        transform=None,
    ):
        super().__init__()
        self.root_dir  = Path(root_dir)
        self.split     = split
        self.transform = transform or get_transforms(split, img_size)

        self.samples: list[tuple[Path, int]] = []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

        for cls_name, cls_id in CLASS_MAP.items():
            cls_dir = self.root_dir / "imgs" / cls_name
            if not cls_dir.exists():
                print(f"[WARNING] Carpeta no encontrada: {cls_dir}")
                continue
            files = [p for p in cls_dir.iterdir() if p.suffix.lower() in exts]
            self.samples.extend([(p, cls_id) for p in sorted(files)])

        counts = [sum(1 for _, l in self.samples if l == i) for i in range(NUM_CLASSES)]
        print(
            f"[Dataset] split={split:<6}  total={len(self.samples):>5}  "
            f"health={counts[0]:>4}  sick={counts[1]:>4}  tb={counts[2]:>4}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img)
        return tensor, label


# ── DataLoaders ───────────────────────────────────────────────────────────────
def get_dataloaders(
    root_dir:    str,
    batch_size:  int   = 32,
    img_size:    int   = 224,
    val_split:   float = 0.15,
    test_split:  float = 0.10,
    num_workers: int   = 4,
    seed:        int   = 42,
):
    """
    Crea train / val / test DataLoaders.
    Usa WeightedRandomSampler para compensar el desbalance TB (minoría).
    """
    full_ds = TBDataset(root_dir, split="train", img_size=img_size)
    n       = len(full_ds)

    rng     = np.random.default_rng(seed)
    indices = rng.permutation(n).tolist()

    n_test  = int(n * test_split)
    n_val   = int(n * val_split)
    n_train = n - n_val - n_test

    train_idx = indices[:n_train]
    val_idx   = indices[n_train: n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    # ── WeightedRandomSampler (balancea TB vs health/sick) ───────────────────
    labels        = [full_ds.samples[i][1] for i in train_idx]
    counts_arr    = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    counts_arr[counts_arr == 0] = 1
    class_weights = 1.0 / counts_arr
    class_weights[2] *= 2.5          # Extra peso a TB (clase minoritaria)
    sample_weights  = [class_weights[l] for l in labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_idx),
        replacement=True,
    )

    val_transform  = get_transforms("val",  img_size)
    test_transform = get_transforms("test", img_size)

    # Subsets con transforms correctos
    train_ds = Subset(full_ds, train_idx)

    val_full = TBDataset(root_dir, split="val", img_size=img_size, transform=val_transform)
    val_ds   = Subset(val_full, val_idx)

    test_full = TBDataset(root_dir, split="test", img_size=img_size, transform=test_transform)
    test_ds   = Subset(test_full, test_idx)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=0,
    )

    print(
        f"[DataLoader] train_batches={len(train_loader)}  "
        f"val_batches={len(val_loader)}  test_samples={len(test_ds)}"
    )
    return train_loader, val_loader, test_loader
