"""
src/metrics.py
Quantitative evaluation metrics and publication-quality Matplotlib visualizers:
- PSNR (Peak Signal-to-Noise Ratio in dB)
- SSIM (Structural Similarity Index Measure)
- BER (Bit Error Rate)
- Evaluation comparison plots (MC-SS vs Single-Image DSSS)
- Residual Heatmap (Spread Spectrum thermal noise visualization)
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import List, Tuple, Dict, Optional, Any
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn


def compute_psnr(image_true: np.ndarray, image_test: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    Higher PSNR (> 40 dB) indicates high visual imperceptibility.
    """
    if image_true.shape != image_test.shape:
        raise ValueError(f"Shape mismatch: {image_true.shape} vs {image_test.shape}")
    
    # Use scikit-image peak_signal_noise_ratio with data_range=255
    return float(psnr_fn(image_true, image_test, data_range=255))


def compute_ssim(image_true: np.ndarray, image_test: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM) in range [-1, 1].
    Values > 0.98 indicate virtually identical structural visual quality.
    """
    if image_true.shape != image_test.shape:
        raise ValueError(f"Shape mismatch: {image_true.shape} vs {image_test.shape}")

    if image_true.ndim == 3 and image_true.shape[2] == 3:
        return float(ssim_fn(image_true, image_test, channel_axis=2, data_range=255))
    else:
        return float(ssim_fn(image_true, image_test, data_range=255))


def compute_ber(bits_orig: np.ndarray, bits_extracted: np.ndarray) -> float:
    """
    Compute Bit Error Rate (BER) between original and recovered bitstreams.
    BER in [0.0, 1.0]. BER = 0.0 indicates 100% error-free reconstruction.
    """
    len_orig = len(bits_orig)
    if len_orig == 0:
        return 0.0

    # Match length
    if len(bits_extracted) < len_orig:
        padded_ext = np.zeros(len_orig, dtype=np.int8)
        padded_ext[: len(bits_extracted)] = bits_extracted
    else:
        padded_ext = bits_extracted[:len_orig]

    errors = np.sum(bits_orig[:len_orig] != padded_ext)
    return float(errors / len_orig)


def plot_residual_heatmap(cover: np.ndarray, stego: np.ndarray, patch_size: int = 50,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot magnified (patch_size x patch_size) residual difference (I_stego - I_cover)
    to visually prove that the spread spectrum signal mimics zero-mean thermal noise.
    """
    if cover.ndim == 3:
        c_gray = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY) if cover.shape[2] == 3 else cover[:, :, 0]
    else:
        c_gray = cover

    if stego.ndim == 3:
        s_gray = cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY) if stego.shape[2] == 3 else stego[:, :, 0]
    else:
        s_gray = stego

    residual = s_gray.astype(np.float32) - c_gray.astype(np.float32)
    
    # Crop central patch
    h, w = residual.shape
    y0 = (h - patch_size) // 2
    x0 = (w - patch_size) // 2
    patch = residual[y0 : y0 + patch_size, x0 : x0 + patch_size]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(patch, cmap='coolwarm', vmin=-3.0, vmax=3.0)
    plt.colorbar(im, ax=ax, label=r'Residual Amplitude ($I_{\mathrm{stego}} - I_{\mathrm{cover}}$)')
    ax.set_title(f"Spread Spectrum Residual ({patch_size}x{patch_size} Patch)\nZero-Mean Pseudo-Noise Behavior",
                 fontsize=11, fontweight='bold')
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig


def plot_psnr_ssim_vs_alpha(alphas: List[float], psnr_values: List[float], ssim_values: List[float],
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Graph 1: Imperceptibility Benchmarking: PSNR & SSIM vs. Embedding Amplitude (alpha).
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:blue'
    ax1.set_xlabel(r'Embedding Amplitude ($\alpha$)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('PSNR (dB)', color=color, fontsize=12, fontweight='bold')
    line1 = ax1.plot(alphas, psnr_values, color=color, marker='o', linewidth=2.2, label='PSNR (dB)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(40.0, color='darkblue', linestyle='--', alpha=0.7, label='40 dB Imperceptibility Threshold')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('SSIM', color=color, fontsize=12, fontweight='bold')
    line2 = ax2.plot(alphas, ssim_values, color=color, marker='s', linewidth=2.2, linestyle='-.', label='SSIM')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([min(0.97, min(ssim_values) - 0.01), 1.002])

    lines = line1 + line2 + [ax1.get_lines()[-1]]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left')

    plt.title(r'Tradeoff: Imperceptibility vs. Embedding Amplitude ($\alpha$)', fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig


def plot_ber_vs_attack(param_name: str, param_values: List[Any], ber_mcss: List[float],
                       ber_dsss: List[float], title: str, save_path: Optional[str] = None) -> plt.Figure:
    """
    Graph 2: BER vs. Attack Intensity (MC-SS with ECC vs. Single-Image DSSS).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(param_values))

    ax.plot(param_values, ber_mcss, marker='o', color='forestgreen', linewidth=2.5,
            label='MC-SS (N-Carriers + Hamming ECC)')
    ax.plot(param_values, ber_dsss, marker='^', color='crimson', linewidth=2.2, linestyle='--',
            label='Single-Image DSSS Baseline')

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel(param_name, fontsize=12, fontweight='bold')
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12, fontweight='bold')
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=11, loc='upper left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig


def plot_ber_vs_carrier_loss(carriers_available: List[int], ber_mcss_ecc: List[float],
                             ber_mcss_no_ecc: List[float], ber_dsss: List[float],
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Graph 3: BER vs. Number of Available Carriers (Stress-testing missing carrier scenario).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(carriers_available, ber_mcss_ecc, marker='o', color='forestgreen', linewidth=2.5,
            label='MC-SS (Hamming ECC + Interleaving)')
    ax.plot(carriers_available, ber_mcss_no_ecc, marker='s', color='orange', linewidth=2.0, linestyle='-.',
            label='MC-SS (No ECC - Graceful Degradation)')
    ax.plot(carriers_available, ber_dsss, marker='X', color='crimson', linewidth=2.2, linestyle=':',
            label='Single-Image DSSS (Catastrophic Loss)')

    ax.set_title('Carrier Loss Resilience: BER vs. Available Carriers', fontsize=13, fontweight='bold')
    ax.set_xlabel('Number of Received Carriers ($N, N-1, N-2, \\dots$)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12, fontweight='bold')
    ax.set_ylim([-0.02, 1.05])
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=10, loc='center right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig
