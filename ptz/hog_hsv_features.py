"""Extractor HOG_HSV_V1 para inferencia, equivalente al script de entrenamiento.

Parámetros de ``extract_hog_hsv_features_v1.py`` y de
``hog_hsv_features_v1/feature_config.json``. No importa pandas ni procesa cachés.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage.feature import hog

FEATURE_VERSION = "HOG_HSV_V1"
FEATURE_DIM = 1812
CANVAS_SIZE = 64
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"
HSV_BINS = (16, 16, 16)


def resize_preserve_aspect(
    image: NDArray[np.uint8], canvas: int = CANVAS_SIZE,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Devuelve contenido escalado y canvas, con la misma geometría original."""
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or image.size == 0):
        raise ValueError("El recorte debe ser una imagen BGR uint8 no vacía (alto, ancho, 3).")
    h, w = image.shape[:2]
    scale = min(canvas / float(w), canvas / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (nw, nh), interpolation=interp)
    top = (canvas - nh) // 2
    bottom = canvas - nh - top
    left = (canvas - nw) // 2
    right = canvas - nw - left
    border = cv2.BORDER_REFLECT_101 if min(nh, nw) > 1 else cv2.BORDER_REPLICATE
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, border)
    if padded.shape[:2] != (canvas, canvas):
        raise ValueError(f"El canvas HOG tiene dimensiones inesperadas: {padded.shape}.")
    return resized, padded


def hsv_histogram(content_bgr: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Tres histogramas de 16 bins, normalizados por canal, antes del padding."""
    hsv = cv2.cvtColor(content_bgr, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [HSV_BINS[0]], [0, 180]).reshape(-1)
    hist_s = cv2.calcHist([hsv], [1], None, [HSV_BINS[1]], [0, 256]).reshape(-1)
    hist_v = cv2.calcHist([hsv], [2], None, [HSV_BINS[2]], [0, 256]).reshape(-1)
    parts = []
    for hist in (hist_h, hist_s, hist_v):
        denom = float(hist.sum())
        if denom > 0:
            hist = hist / denom
        parts.append(hist.astype(np.float32, copy=False))
    return np.concatenate(parts).astype(np.float32, copy=False)


def extract_features(image_bgr: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Convierte un recorte BGR en el vector HOG, H, S, V del entrenamiento."""
    content, canvas = resize_preserve_aspect(image_bgr, CANVAS_SIZE)
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    hog_vec = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        visualize=False,
        feature_vector=True,
        transform_sqrt=False,
    ).astype(np.float32, copy=False)
    color_vec = hsv_histogram(content)
    feat = np.concatenate([hog_vec, color_vec]).astype(np.float32, copy=False)
    if feat.shape != (FEATURE_DIM,) or not np.isfinite(feat).all():
        raise ValueError("El extractor HOG_HSV_V1 devolvió características inválidas.")
    return feat
