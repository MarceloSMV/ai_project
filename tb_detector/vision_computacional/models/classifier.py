"""
models/classifier.py — Clasificador de Tuberculosis con EfficientNet-B3 / ResNet-50

Arquitectura:
  - Backbone: EfficientNet-B3 (por defecto) o ResNet-50, preentrenados en ImageNet.
  - Cabeza: Dropout + capa lineal → 3 clases (health, sick, tb).
  - Grad-CAM: Extrae mapas de calor de la última capa convolucional para
              generar bounding boxes dinámicos en inferencia.

Uso:
    model = build_model(backbone="efficientnet")
    model = build_model(backbone="resnet50")
"""

from __future__ import annotations
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B3_Weights,
    ResNet50_Weights,
)

# ── Constantes ────────────────────────────────────────────────────────────────
CLASS_NAMES = ["health", "sick", "tb"]
NUM_CLASSES  = 3

# Colores BGR por clase para visualización
CLASS_COLORS = {
    0: (0,   200,  50),   # health  → verde
    1: (0,   165, 255),   # sick    → naranja
    2: (0,    0,  255),   # tb      → rojo
}


# ── Construcción del modelo ───────────────────────────────────────────────────
def build_model(
    num_classes: int = NUM_CLASSES,
    backbone:    str = "efficientnet",
    dropout:     float = 0.40,
    pretrained:  bool = True,
) -> nn.Module:
    """
    Retorna un clasificador preentrenado adaptado para 3 clases.

    Args:
        backbone:  'efficientnet' (B3) o 'resnet50'.
        dropout:   Tasa de dropout en la cabeza de clasificación.
        pretrained: Usar pesos ImageNet (True) o aleatorios (False).
    """
    if backbone == "efficientnet":
        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model   = models.efficientnet_b3(weights=weights)
        in_feat = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_feat, num_classes),
        )
        # pyrefly: ignore [bad-argument-type]
        model._backbone_name = "efficientnet"
    else:  # resnet50
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model   = models.resnet50(weights=weights)
        in_feat = model.fc.in_features
        # setattr evita el error de tipado estático:
        # model.fc está anotado como nn.Linear en torchvision, pero en
        # Python el reemplazo dinámico con nn.Sequential funciona correctamente.
        setattr(model, "fc", nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_feat, num_classes),
        ))
        # pyrefly: ignore [bad-argument-type]
        model._backbone_name = "resnet50"

    return model


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM sobre la última capa convolucional del backbone.

    Genera un mapa de calor [H, W] normalizado en [0, 1] que indica
    qué zonas de la imagen activaron la decisión del clasificador.

    Uso:
        cam      = GradCAM(model)
        heatmap  = cam.generate(img_tensor, target_class=2)  # 2=TB
        overlay  = GradCAM.overlay(img_bgr, heatmap)
        boxes    = GradCAM.to_boxes(heatmap, threshold=0.55)
    """

    def __init__(self, model: nn.Module):
        self.model       = model
        self.activations: torch.Tensor | None = None
        self.gradients:   torch.Tensor | None = None
        self._hooks: list = []
        self._register_hooks()

    def _get_target_layer(self) -> nn.Module:
        name = getattr(self.model, "_backbone_name", "efficientnet")
        if name == "efficientnet":
            # Último bloque MBConv antes del avg pool
            return self.model.features[-1]
        else:  # resnet50
            return self.model.layer4[-1].conv3

    def _register_hooks(self):
        target = self._get_target_layer()

        def fwd(module, inp, out):
            self.activations = out.detach()

        def bwd(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hooks.append(target.register_forward_hook(fwd))
        self._hooks.append(target.register_full_backward_hook(bwd))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.enable_grad()
    def generate(
        self,
        img_tensor:   torch.Tensor,
        target_class: int,
        device:       str = "cpu",
    ) -> np.ndarray:
        """
        Computa el mapa de calor Grad-CAM para `target_class`.

        Returns:
            heatmap: np.float32 [H, W], valores en [0, 1].
        """
        self.model.eval()
        tensor = img_tensor.to(device).requires_grad_(True)

        logits = self.model(tensor)             # [1, NUM_CLASSES]
        score  = logits[0, target_class]
        self.model.zero_grad()
        score.backward()

        grads = self.gradients   # [1, C, h, w]
        acts  = self.activations  # [1, C, h, w]

        if grads is None or acts is None:
            raise RuntimeError("Grad-CAM: los hooks no capturaron tensores.")

        weights = grads.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
        cam     = (weights * acts).sum(dim=1).squeeze()   # [h, w]
        cam     = torch.relu(cam).cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        _, _, H, W = img_tensor.shape
        cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
        return cam.astype(np.float32)

    @staticmethod
    def overlay(
        img_bgr:  np.ndarray,
        heatmap:  np.ndarray,
        alpha:    float = 0.45,
        colormap: int   = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Superpone el mapa de calor sobre la imagen original."""
        hm_u8 = (heatmap * 255).astype(np.uint8)
        hm_c  = cv2.applyColorMap(hm_u8, colormap)
        hm_c  = cv2.resize(hm_c, (img_bgr.shape[1], img_bgr.shape[0]))
        return cv2.addWeighted(img_bgr, 1 - alpha, hm_c, alpha, 0)

    @staticmethod
    def to_boxes(
        heatmap:    np.ndarray,
        threshold:  float = 0.55,
        min_area:   int   = 1200,
        max_boxes:  int   = 4,
    ) -> list[dict]:
        """
        Convierte un mapa de calor en bounding boxes detectando contornos.

        Args:
            threshold:  Fracción de activación máxima para binarizar (0–1).
            min_area:   Área mínima de contorno (filtra ruido de fondo).
            max_boxes:  Máximo de cajas a retornar (las de mayor área).

        Returns:
            Lista de dicts con keys: x1, y1, x2, y2, confidence (0–1).
        """
        binary = (heatmap >= threshold).astype(np.uint8) * 255
        # Morfología: cerrar huecos pequeños y suavizar contornos
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Confianza proporcional a la activación media dentro del contorno
            mask = np.zeros_like(heatmap, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            confidence = float(heatmap[mask > 0].mean())
            boxes.append({"x1": x, "y1": y, "x2": x + w, "y2": y + h,
                          "confidence": round(confidence, 4), "area": area})

        # Ordenar por área descendente y limitar cantidad
        boxes.sort(key=lambda b: b["area"], reverse=True)
        return boxes[:max_boxes]


# ── Visualización ─────────────────────────────────────────────────────────────
def draw_boxes(
    img_bgr:    np.ndarray,
    boxes:      list[dict],
    class_id:   int,
    label_text: str = "",
) -> np.ndarray:
    """
    Dibuja las cajas delimitadoras derivadas de Grad-CAM sobre la imagen.

    Args:
        boxes:      Lista de dicts {x1, y1, x2, y2, confidence}.
        class_id:   Índice de clase predicha (0=health, 1=sick, 2=tb).
        label_text: Texto adicional en el label (ej: "TB 94.2%").
    """
    vis   = img_bgr.copy()
    color = CLASS_COLORS.get(class_id, (200, 200, 200))
    font  = cv2.FONT_HERSHEY_SIMPLEX

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        conf_pct = box["confidence"] * 100

        thickness = 3 if class_id == 2 else 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        txt = label_text if i == 0 else f"zona {i+1} ({conf_pct:.0f}%)"
        (tw, th), bl = cv2.getTextSize(txt, font, 0.55, 1)
        cv2.rectangle(vis, (x1, y1 - th - bl - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, txt, (x1 + 2, y1 - bl - 2),
                    font, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return vis
