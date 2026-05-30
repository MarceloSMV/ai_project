"""
utils/inference.py — Inferencia standalone para TBDetector (Clasificador + Grad-CAM)

El clasificador predice la clase (health / sick / tb) y usa Grad-CAM para
localizar dinámicamente las zonas anómalas y generar bounding boxes sin
necesidad de anotaciones manuales.

Uso como librería (Flask/FastAPI):
    infer  = TBInference("runs/checkpoints/best.pt")
    result = infer.predict_bytes(img_bytes)
    # result["verdict"]       → "health" | "sick" | "tb"
    # result["class_report"]  → {"health": 5.1, "sick": 2.3, "tb": 92.6}
    # result["vis_boxes"]     → imagen BGR con bounding boxes
    # result["vis_gradcam"]   → imagen BGR con Grad-CAM superpuesto

Uso por línea de comandos:
    python -m utils.inference --img rx.jpg
    python -m utils.inference --img rx.jpg --model runs/checkpoints/best.pt --save out.jpg
"""

from __future__ import annotations
import argparse
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# Importar desde models/classifier.py (mismo paquete)
from models.classifier import (
    build_model,
    GradCAM,
    draw_boxes,
    CLASS_NAMES,
    NUM_CLASSES,
    CLASS_COLORS,
)

IMG_SIZE  = 224
TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class TBInference:
    """
    Clase de inferencia reutilizable para el clasificador de TB.
    Compatible con Flask/FastAPI y uso standalone.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        backbone:        str = "efficientnet",
        device:          str | None = None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.backbone = backbone
        self.model    = self._load(checkpoint_path)
        self.grad_cam = GradCAM(self.model)

    def _load(self, ckpt_path: str | None) -> torch.nn.Module:
        # Intentar detectar backbone desde el checkpoint
        backbone = self.backbone
        if ckpt_path and Path(ckpt_path).exists():
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            saved_cfg = ckpt.get("cfg", {})
            backbone  = saved_cfg.get("backbone", backbone)
            self.backbone = backbone

        model = build_model(num_classes=NUM_CLASSES, backbone=backbone, pretrained=False)

        if ckpt_path and Path(ckpt_path).exists():
            state = ckpt.get("model", ckpt)
            model.load_state_dict(state)
            acc = ckpt.get("best_acc", ckpt.get("val_acc", 0.0))
            print(f"[Inference] Modelo {backbone} cargado: {ckpt_path}")
            print(f"[Inference] Mejor Accuracy registrada: {acc*100:.2f}%")
        else:
            print("[Inference] AVISO: sin checkpoint, pesos aleatorios (solo para desarrollo).")

        model.to(self.device).eval()
        return model

    def preprocess(self, img_path: str) -> tuple[np.ndarray, torch.Tensor]:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"No se encontró: {img_path}")
        img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        tensor  = TRANSFORM(pil_img).unsqueeze(0)
        return img_bgr, tensor

    def preprocess_bytes(self, img_bytes: bytes) -> tuple[np.ndarray, torch.Tensor]:
        """Versión que acepta bytes directamente (sin archivo temporal)."""
        arr    = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("No se pudo decodificar la imagen desde bytes.")
        img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        tensor  = TRANSFORM(pil_img).unsqueeze(0)
        return img_bgr, tensor

    @torch.no_grad()
    def _classify(self, tensor: torch.Tensor) -> tuple[int, np.ndarray]:
        """Retorna (clase_predicha, probabilidades[NUM_CLASSES])."""
        logits = self.model(tensor.to(self.device))
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred   = int(probs.argmax())
        return pred, probs

    def predict(
        self,
        img_path:        str,
        score_thresh:    float = 0.55,   # umbral Grad-CAM para generar cajas
        include_gradcam: bool  = True,
    ) -> dict:
        """
        Predice la clase de la radiografía y genera la visualización.

        Returns dict con:
            verdict:       str → 'health' | 'sick' | 'tb'
            class_report:  dict → {'health': 5.1, 'sick': 2.3, 'tb': 92.6}
            vis_boxes:     np.ndarray BGR con bounding boxes Grad-CAM
            vis_gradcam:   np.ndarray BGR con overlay de calor
            inference_ms:  float
            boxes:         list[dict] con {x1,y1,x2,y2,confidence}
        """
        img_bgr, tensor = self.preprocess(img_path)

        t0              = time.perf_counter()
        pred_class, probs = self._classify(tensor)
        ms              = (time.perf_counter() - t0) * 1000

        verdict = CLASS_NAMES[pred_class]
        class_report = {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
                        for i in range(NUM_CLASSES)}

        vis_boxes   = img_bgr.copy()
        vis_gradcam = None
        boxes       = []

        if include_gradcam:
            try:
                heatmap     = self.grad_cam.generate(
                    tensor, target_class=pred_class, device=str(self.device)
                )
                vis_gradcam = GradCAM.overlay(img_bgr.copy(), heatmap)

                # Solo dibujar cajas si hay anomalía (sick o tb)
                if pred_class > 0:
                    boxes = GradCAM.to_boxes(heatmap, threshold=score_thresh)
                    if boxes:
                        conf_pct = probs[pred_class] * 100
                        label    = f"{verdict.upper()} {conf_pct:.1f}%"
                        vis_boxes   = draw_boxes(img_bgr.copy(), boxes, pred_class, label)
                        vis_gradcam = draw_boxes(vis_gradcam,    boxes, pred_class, label)
                    else:
                        # Fallback: dibujar texto de diagnóstico si no hay contornos grandes
                        self._draw_label(vis_boxes, verdict, pred_class, probs[pred_class])
                else:
                    self._draw_label(vis_boxes, verdict, pred_class, probs[pred_class])

            except Exception as e:
                print(f"[Inference] Grad-CAM falló: {e}")
                self._draw_label(vis_boxes, verdict, pred_class, probs[pred_class])
        else:
            self._draw_label(vis_boxes, verdict, pred_class, probs[pred_class])

        return {
            "verdict":      verdict,
            "class_report": class_report,
            "vis_boxes":    vis_boxes,
            "vis_gradcam":  vis_gradcam,
            "boxes":        boxes,
            "inference_ms": round(ms, 2),
        }

    def predict_bytes(self, img_bytes: bytes, **kwargs) -> dict:
        """Variante directa para Flask (sin archivo temporal)."""
        img_bgr, tensor = self.preprocess_bytes(img_bytes)

        score_thresh    = kwargs.pop("score_thresh", 0.55)
        include_gradcam = kwargs.pop("include_gradcam", True)

        t0              = time.perf_counter()
        pred_class, probs = self._classify(tensor)
        ms              = (time.perf_counter() - t0) * 1000

        verdict      = CLASS_NAMES[pred_class]
        class_report = {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
                        for i in range(NUM_CLASSES)}

        vis_boxes   = img_bgr.copy()
        vis_gradcam = None
        boxes       = []

        if include_gradcam:
            try:
                heatmap     = self.grad_cam.generate(
                    tensor, target_class=pred_class, device=str(self.device)
                )
                vis_gradcam = GradCAM.overlay(img_bgr.copy(), heatmap)

                if pred_class > 0:
                    boxes = GradCAM.to_boxes(heatmap, threshold=score_thresh)
                    if boxes:
                        conf_pct = probs[pred_class] * 100
                        label    = f"{verdict.upper()} {conf_pct:.1f}%"
                        vis_boxes   = draw_boxes(img_bgr.copy(), boxes, pred_class, label)
                        vis_gradcam = draw_boxes(vis_gradcam,    boxes, pred_class, label)
                    else:
                        self._draw_label(vis_boxes, verdict, pred_class, probs[pred_class])
                else:
                    self._draw_label(vis_boxes, verdict, pred_class, probs[pred_class])

            except Exception as e:
                print(f"[Inference] Grad-CAM falló: {e}")
                self._draw_label(vis_boxes, verdict, pred_class, probs[pred_class])
        else:
            self._draw_label(vis_boxes, verdict, pred_class, probs[pred_class])

        return {
            "verdict":      verdict,
            "class_report": class_report,
            "vis_boxes":    vis_boxes,
            "vis_gradcam":  vis_gradcam,
            "boxes":        boxes,
            "inference_ms": round(ms, 2),
        }

    @staticmethod
    def _draw_label(img: np.ndarray, verdict: str, class_id: int, prob: float):
        """Dibuja solo el texto de diagnóstico cuando no hay cajas."""
        color    = CLASS_COLORS.get(class_id, (200, 200, 200))
        text     = f"{verdict.upper()}  {prob*100:.1f}%"
        font     = cv2.FONT_HERSHEY_SIMPLEX
        H, W     = img.shape[:2]
        (tw, th), bl = cv2.getTextSize(text, font, 0.8, 2)
        cv2.rectangle(img, (0, H - th - bl - 14), (tw + 10, H), color, -1)
        cv2.putText(img, text, (5, H - bl - 8), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)


# ── CLI ───────────────────────────────────────────────────────────────────────
def _cli():
    parser = argparse.ArgumentParser(description="TBDetector — Inferencia standalone")
    parser.add_argument("--img",      required=True,  help="Ruta a la radiografía")
    parser.add_argument("--model",    default=None,   help="Checkpoint .pt")
    parser.add_argument("--backbone", default="efficientnet", choices=["efficientnet", "resnet50"])
    parser.add_argument("--thresh",   type=float, default=0.55)
    parser.add_argument("--save",     default=None)
    parser.add_argument("--no-show",  action="store_true")
    args = parser.parse_args()

    if args.model is None:
        _here     = Path(__file__).resolve().parent.parent
        args.model = str(_here / "runs" / "checkpoints" / "best.pt")

    infer  = TBInference(args.model, backbone=args.backbone)
    result = infer.predict(args.img, score_thresh=args.thresh)

    print(f"\n── Resultado ──────────────────────────────")
    print(f"  Veredicto  : {result['verdict'].upper()}")
    print(f"  Tiempo     : {result['inference_ms']} ms")
    print(f"  Confianzas :")
    for cls, score in result["class_report"].items():
        bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
        print(f"    {cls:8s} {bar} {score:.1f}%")
    print(f"  Cajas Grad-CAM: {len(result['boxes'])}")

    if args.save:
        cv2.imwrite(args.save, result["vis_boxes"])
        print(f"  Guardado en: {args.save}")
        if result["vis_gradcam"] is not None:
            p = Path(args.save)
            gcam = str(p.parent / (p.stem + "_gradcam" + p.suffix))
            cv2.imwrite(gcam, result["vis_gradcam"])
            print(f"  Grad-CAM en: {gcam}")

    if not args.no_show:
        cv2.imshow("TBDetector — Diagnóstico", result["vis_boxes"])
        if result["vis_gradcam"] is not None:
            cv2.imshow("TBDetector — Grad-CAM", result["vis_gradcam"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _cli()
