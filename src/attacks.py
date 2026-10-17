"""
src/attacks.py
Simulated communication channel attacks and image distortions:
- JPEG Compression (varying quality factors QF)
- Additive White Gaussian Noise (AWGN with varying sigma)
- Spatial Cropping (zeroing out central or edge region)
- Resizing / Downsampling-Upsampling
- Carrier Drop / Missing Carrier Channel Simulation
- Salt & Pepper Noise and Gaussian Blur
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple


def apply_jpeg_compression(image: np.ndarray, quality_factor: int = 75) -> np.ndarray:
    """
    Simulate lossy JPEG compression on an image.
    quality_factor: [10, 100], where 100 is best quality / lowest compression.
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality_factor)]
    result, enc_img = cv2.imencode('.jpg', image, encode_param)
    if not result:
        raise RuntimeError("JPEG encoding failed.")
    dec_img = cv2.imdecode(enc_img, cv2.IMREAD_UNCHANGED)
    return dec_img


def apply_gaussian_noise(image: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """
    Add zero-mean Gaussian noise with standard deviation sigma to the image.
    """
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return np.round(noisy).astype(np.uint8)


def apply_cropping(image: np.ndarray, crop_fraction: float = 0.20) -> np.ndarray:
    """
    Simulate spatial cropping / packet loss by zeroing out a centered rectangular patch
    occupying crop_fraction of the total area.
    """
    h, w = image.shape[:2]
    side_fraction = np.sqrt(crop_fraction)
    ch = int(h * side_fraction)
    cw = int(w * side_fraction)
    
    y1 = (h - ch) // 2
    y2 = y1 + ch
    x1 = (w - cw) // 2
    x2 = x1 + cw
    
    attacked = image.copy()
    attacked[y1:y2, x1:x2] = 0
    return attacked


def apply_resizing(image: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """
    Simulate downsampling and upsampling back to original resolution.
    """
    h, w = image.shape[:2]
    down_w = max(1, int(w * scale))
    down_h = max(1, int(h * scale))
    downscaled = cv2.resize(image, (down_w, down_h), interpolation=cv2.INTER_AREA)
    upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_LINEAR)
    return upscaled


def apply_salt_pepper_noise(image: np.ndarray, amount: float = 0.02) -> np.ndarray:
    """
    Add impulsive Salt & Pepper noise.
    """
    out = image.copy()
    num_salt = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[tuple(coords)] = 255
    
    num_pepper = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[tuple(coords)] = 0
    return out


def apply_gaussian_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Apply low-pass Gaussian smoothing filter.
    """
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(image, (k, k), 0)


def simulate_carrier_drop(stegos: List[np.ndarray], dropped_indices: List[int]) -> List[Optional[np.ndarray]]:
    """
    Simulate intercepted, corrupted, or completely dropped carrier images in MC-SS.
    Returns list where dropped carriers are replaced with None (or black zeros).
    """
    attacked_stegos = []
    for k, stego in enumerate(stegos):
        if k in dropped_indices:
            # Dropped carrier
            attacked_stegos.append(None)
        else:
            attacked_stegos.append(stego.copy())
    return attacked_stegos
