"""Canonical image geometry shared by NERO recording and policy clients."""

from __future__ import annotations

import cv2
import numpy as np


MODEL_IMAGE_SIZE = 224


def rotate_external_image(image: np.ndarray) -> np.ndarray:
    """Rotate the fixed external camera 90 degrees counterclockwise."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 external image, got {image.shape}")
    return np.ascontiguousarray(np.rot90(image, k=1))


def resize_with_pad(image: np.ndarray, size: int = MODEL_IMAGE_SIZE) -> np.ndarray:
    """Match OpenPI's linear resize-with-black-padding geometry."""
    height, width = image.shape[:2]
    ratio = max(width / size, height / size)
    resized_height = int(height / ratio)
    resized_width = int(width / ratio)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    output = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - resized_height) // 2
    left = (size - resized_width) // 2
    output[top : top + resized_height, left : left + resized_width] = resized
    return np.ascontiguousarray(output)


def prepare_external_model_image(image: np.ndarray, size: int = MODEL_IMAGE_SIZE) -> np.ndarray:
    return resize_with_pad(rotate_external_image(image), size=size)
