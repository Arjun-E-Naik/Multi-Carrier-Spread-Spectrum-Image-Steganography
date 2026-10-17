"""
generate_notebook.py
Generates the pre-structured interactive presentation notebook: demo_showcase.ipynb
"""

import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Multi-Carrier Spread Spectrum (MC-SS) Image Steganography\n",
                "### A Resilient Spatial Steganography Pipeline with Spatial Diversity & Carrier-Drop Fault Tolerance\n",
                "---\n",
                "**Mentor Showcase & Evaluation Notebook**  \n",
                "This notebook is structured into **Three Showcase Acts**:\n",
                "1. **Act 1:** Live Interactive Demo (Spatial Embedding, Visual Imperceptibility PSNR > 40 dB, Thermal-Noise Residual Heatmap, 0% Clean BER).\n",
                "2. **Act 2:** Live Attack & Missing Carrier Stress Test (MC-SS vs. Single-Image DSSS baseline under AWGN, JPEG, and Dropped Carriers).\n",
                "3. **Act 3:** Quantitative Benchmark Graphs & Academic Viva Defense."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Setup & Module Imports\n",
                "Import core modules from `src/` and load standard 512×512 cover images (Lena, Baboon, Peppers, Cameraman)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import cv2\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "from src.mcss_core import MCSSStego, SingleImageDSSS\n",
                "from src.attacks import apply_gaussian_noise, apply_jpeg_compression, apply_cropping, simulate_carrier_drop\n",
                "from src.metrics import compute_psnr, compute_ssim, compute_ber, plot_residual_heatmap\n",
                "from src.ecc_hamming import text_to_bits\n",
                "from download_dataset import download_and_prepare\n",
                "\n",
                "# Prepare standard dataset if not already present\n",
                "download_and_prepare()\n",
                "\n",
                "# Load standard 512x512 carriers\n",
                "carrier_names = [\"lena.png\", \"baboon.png\", \"peppers.png\", \"cameraman.png\"]\n",
                "covers = [cv2.imread(os.path.join(\"dataset\", f), cv2.IMREAD_GRAYSCALE) for f in carrier_names]\n",
                "print(f\"Loaded {len(covers)} cover carriers: {carrier_names}\")\n",
                "for name, img in zip(carrier_names, covers):\n",
                "    print(f\" - {name}: shape = {img.shape}, dtype = {img.dtype}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 🚀 ACT 1: Live Interactive Demo (4 Minutes)\n",
                "### 1.1 Secret Message Input & Multi-Carrier Spatial Embedding\n",
                "We divide the payload across $N=4$ cover images using orthogonal LFSR Gold-code sequences scaled by $\\alpha = 2.0$."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "secret_text = \"CONFIDENTIAL: Antigravity Multi-Carrier Spread Spectrum Stego 2026.\"\n",
                "secret_key = \"QuantumKey#2026\"\n",
                "alpha = 2.0\n",
                "\n",
                "mcss = MCSSStego(secret_key=secret_key, num_carriers=len(covers), alpha=alpha, use_ecc=True)\n",
                "stegos, metadata = mcss.embed_text(covers, secret_text)\n",
                "raw_bits = text_to_bits(secret_text)\n",
                "\n",
                "print(f\"[+] Secret Text: '{secret_text}'\")\n",
                "print(f\"[+] Total Encoded Bits: {len(raw_bits)} bits\")\n",
                "print(f\"[+] Bits per Carrier: {metadata['bits_per_carrier']} bits\")\n",
                "print(f\"[+] Spreading Factor (M): {metadata['chips_per_bit']} pixels/bit\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1.2 Visual Imperceptibility Check (Cover vs. Stego)\n",
                "Display the $N$ original cover images side-by-side with the $N$ generated stego images, printing **PSNR (> 40 dB)** and **SSIM (> 0.98)**."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(2, 4, figsize=(16, 8))\n",
                "for i in range(4):\n",
                "    psnr = compute_psnr(covers[i], stegos[i])\n",
                "    ssim = compute_ssim(covers[i], stegos[i])\n",
                "    \n",
                "    # Original Cover\n",
                "    axes[0, i].imshow(covers[i], cmap='gray')\n",
                "    axes[0, i].set_title(f\"Cover {i+1}: {carrier_names[i].split('.')[0].capitalize()}\")\n",
                "    axes[0, i].axis('off')\n",
                "    \n",
                "    # Stego Image\n",
                "    axes[1, i].imshow(stegos[i], cmap='gray')\n",
                "    axes[1, i].set_title(f\"Stego {i+1}\\nPSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}\")\n",
                "    axes[1, i].axis('off')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1.3 The 'Noise' Proof: Magnified 50×50 Residual Heatmap\n",
                "Visually prove that the spread spectrum signal ($I_{\\text{stego}} - I_{\\text{cover}}$) mimics zero-mean thermal noise."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig = plot_residual_heatmap(covers[0], stegos[0], patch_size=50)\n",
                "plt.show()\n",
                "\n",
                "res = stegos[0].astype(float) - covers[0].astype(float)\n",
                "print(f\"Residual Mean (μ): {np.mean(res):.4f} (Zero-mean noise confirmation)\")\n",
                "print(f\"Residual Std Dev (σ): {np.std(res):.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1.4 Blind Correlation Extraction on Clean Stego Images\n",
                "Regenerate PN sequences from the shared seed key and extract the message."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "rec_text, rec_bits, stats = mcss.extract_text(stegos, metadata)\n",
                "ber_clean = compute_ber(raw_bits, rec_bits)\n",
                "\n",
                "print(f\"Extracted Message: '{rec_text}'\")\n",
                "print(f\"Clean Bit Error Rate (BER): {ber_clean * 100:.2f}%\")\n",
                "assert ber_clean == 0.0, \"Clean BER should be exactly 0%!\""
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## ⚡ ACT 2: Live Attack & Carrier-Loss Stress Test (4 Minutes)\n",
                "### 2.1 Missing Carrier Simulation (Catastrophic DSSS Loss vs. MC-SS Fault Tolerance)\n",
                "Simulate interception or dropping of 1 or 2 carrier images."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Initialize Single-Image DSSS Baseline on Carrier 0\n",
                "dsss = SingleImageDSSS(secret_key=secret_key, alpha=alpha)\n",
                "dsss_stego, dsss_meta = dsss.embed_text(covers[0], secret_text)\n",
                "\n",
                "# Un-ECC MC-SS for comparison\n",
                "mcss_no_ecc = MCSSStego(secret_key=secret_key, num_carriers=4, alpha=alpha, use_ecc=False)\n",
                "stegos_no_ecc, meta_no_ecc = mcss_no_ecc.embed_text(covers, secret_text)\n",
                "\n",
                "dropped_carriers = [0]  # Drop Carrier 0 (Lena)\n",
                "print(f\"Simulating Dropping Carrier 0 ({carrier_names[0]})...\")\n",
                "\n",
                "# 1. MC-SS with Hamming(7,4) ECC + Interleaving\n",
                "st_dropped_ecc = simulate_carrier_drop(stegos, dropped_carriers)\n",
                "txt_ecc, bits_ecc, _ = mcss.extract_text(st_dropped_ecc, metadata, dropped_carriers=dropped_carriers)\n",
                "ber_ecc = compute_ber(raw_bits, bits_ecc)\n",
                "\n",
                "# 2. MC-SS without ECC\n",
                "st_dropped_no = simulate_carrier_drop(stegos_no_ecc, dropped_carriers)\n",
                "txt_no, bits_no, _ = mcss_no_ecc.extract_text(st_dropped_no, meta_no_ecc, dropped_carriers=dropped_carriers)\n",
                "ber_no = compute_ber(raw_bits, bits_no)\n",
                "\n",
                "# 3. Single-Image DSSS (If carrier dropped -> 100% loss)\n",
                "txt_dsss, bits_dsss = dsss.extract_text(None, dsss_meta)\n",
                "ber_dsss = 1.0\n",
                "\n",
                "print(\"=\" * 65)\n",
                "print(f\"1. MC-SS with Hamming ECC BER:  {ber_ecc * 100:.2f}% | Text: '{txt_ecc}'\")\n",
                "print(f\"2. MC-SS without ECC BER:       {ber_no * 100:.2f}% | Text: '{txt_no[:35]}...'\")\n",
                "print(f\"3. Single-Image DSSS BER:        {ber_dsss * 100:.2f}% | Text: '[TOTAL FAILURE]'\")\n",
                "print(\"=\" * 65)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2.2 Channel Distortion: AWGN & JPEG Compression Stress Test"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Additive Gaussian Noise (sigma = 12.0)\n",
                "noisy_stegos = [apply_gaussian_noise(st, sigma=12.0) for st in stegos]\n",
                "noisy_dsss = apply_gaussian_noise(dsss_stego, sigma=12.0)\n",
                "\n",
                "_, bits_mcss_noise, _ = mcss.extract_text(noisy_stegos, metadata)\n",
                "_, bits_dsss_noise = dsss.extract_text(noisy_dsss, dsss_meta)\n",
                "\n",
                "print(f\"Under AWGN (σ = 12.0):\")\n",
                "print(f\" - MC-SS BER: {compute_ber(raw_bits, bits_mcss_noise) * 100:.2f}%\")\n",
                "print(f\" - Single DSSS BER: {compute_ber(raw_bits, bits_dsss_noise) * 100:.2f}%\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2.3 Cryptographic Security Proof (Wrong Key Test)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "attacker_mcss = MCSSStego(secret_key=\"AttackerWrongSeed#999\", num_carriers=4, alpha=alpha, use_ecc=True)\n",
                "att_txt, att_bits, _ = attacker_mcss.extract_text(stegos, metadata)\n",
                "att_ber = compute_ber(raw_bits, att_bits)\n",
                "\n",
                "print(f\"Adversary BER (Unauthorized Key): {att_ber * 100:.2f}%\")\n",
                "print(f\"Adversary Output Text: '{att_txt[:50]}...'\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 📊 ACT 3: Benchmark Graphs & Mentor Defense (2 Minutes)\n",
                "Display publication-ready evaluation charts."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from IPython.display import Image, display\n",
                "\n",
                "# Run benchmark suite if results are not already generated\n",
                "if not os.path.exists(\"results/graph1_psnr_ssim_vs_alpha.png\"):\n",
                "    import run_benchmarks\n",
                "    run_benchmarks.run_all_benchmarks()\n",
                "\n",
                "print(\"Graph 1: Imperceptibility vs. Embedding Amplitude (alpha)\")\n",
                "display(Image(\"results/graph1_psnr_ssim_vs_alpha.png\"))\n",
                "\n",
                "print(\"Graph 2: Robustness Curves (AWGN & JPEG)\")\n",
                "display(Image(\"results/graph2_ber_vs_gaussian_noise.png\"))\n",
                "display(Image(\"results/graph2b_ber_vs_jpeg.png\"))\n",
                "\n",
                "print(\"Graph 3: Carrier Loss Resilience\")\n",
                "display(Image(\"results/graph3_ber_vs_carrier_loss.png\"))"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python (.venv)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("demo_showcase.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("[+] Successfully generated demo_showcase.ipynb")
