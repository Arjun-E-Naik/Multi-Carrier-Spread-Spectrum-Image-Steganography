"""
run_benchmarks.py
Executes the comprehensive quantitative benchmark suite:
- Graph 1: PSNR & SSIM across embedding amplitudes (alpha)
- Graph 2: BER vs. Noise/Compression intensity (MC-SS vs Single-Image DSSS)
- Graph 3: BER vs. Number of Available Carriers (Carrier Drop Stress Test)
- Generates 50x50 residual heatmap proving thermal-noise behavior.
Saves all figures and summary metrics into `results/`.
"""

import os
import sys
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.mcss_core import MCSSStego, SingleImageDSSS
from src.attacks import (
    apply_gaussian_noise,
    apply_jpeg_compression,
    apply_cropping,
    apply_resizing,
    simulate_carrier_drop
)
from src.metrics import (
    compute_psnr,
    compute_ssim,
    compute_ber,
    plot_psnr_ssim_vs_alpha,
    plot_ber_vs_attack,
    plot_ber_vs_carrier_loss,
    plot_residual_heatmap
)
from src.ecc_hamming import text_to_bits
from download_dataset import download_and_prepare

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_cover_images(num_carriers: int = 4) -> list:
    """Load and return standard 512x512 cover images."""
    download_and_prepare()
    candidate_names = ["lena.png", "baboon.png", "peppers.png", "cameraman.png", "barbara.png"]
    covers = []
    for name in candidate_names[:num_carriers]:
        p = os.path.join(DATASET_DIR, name)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load {p}")
        if img.shape != (512, 512):
            img = cv2.resize(img, (512, 512))
        covers.append(img)
    return covers


def run_all_benchmarks():
    print("=" * 70)
    print(" MULTI-CARRIER SPREAD SPECTRUM (MC-SS) BENCHMARK SUITE")
    print("=" * 70)

    num_carriers = 4
    covers = load_cover_images(num_carriers)
    secret_text = "CONFIDENTIAL: Antigravity Multi-Carrier Spread Spectrum Steganography 2026."
    secret_key = "DefenseResearchSeed#9821"
    raw_bits = text_to_bits(secret_text)

    print(f"\n[+] Secret payload: '{secret_text}'")
    print(f"[+] Total payload bits (with 32-bit header): {len(raw_bits)} bits")
    print(f"[+] Cover images: {num_carriers} carriers (512x512 resolution)")

    # -------------------------------------------------------------------------
    # 1. Clean Stego Baseline Verification
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("STEP 1: Clean Transmission Verification (MC-SS)")
    print("-" * 50)
    mcss = MCSSStego(secret_key=secret_key, num_carriers=num_carriers, alpha=2.0, use_ecc=True)
    stegos, metadata = mcss.embed_text(covers, secret_text)
    rec_text, rec_bits, stats = mcss.extract_text(stegos, metadata)
    ber_clean = compute_ber(raw_bits, rec_bits)

    psnrs = [compute_psnr(covers[k], stegos[k]) for k in range(num_carriers)]
    ssims = [compute_ssim(covers[k], stegos[k]) for k in range(num_carriers)]

    print(f"Clean MC-SS Extraction Result: '{rec_text}'")
    print(f"Clean MC-SS BER: {ber_clean * 100:.2f}%")
    for k in range(num_carriers):
        print(f"Carrier {k}: PSNR = {psnrs[k]:.2f} dB, SSIM = {ssims[k]:.4f}")

    # Generate Residual Heatmap for Carrier 0
    heatmap_path = os.path.join(RESULTS_DIR, "residual_heatmap.png")
    plot_residual_heatmap(covers[0], stegos[0], patch_size=50, save_path=heatmap_path)
    plt.close()
    print(f"[+] Saved residual noise heatmap to '{heatmap_path}'")

    # -------------------------------------------------------------------------
    # 2. Graph 1: Imperceptibility vs. Embedding Amplitude (alpha)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("STEP 2: Evaluating Imperceptibility vs. Alpha (Graph 1)")
    print("-" * 50)
    alphas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    avg_psnr_list = []
    avg_ssim_list = []

    for a in alphas:
        engine = MCSSStego(secret_key=secret_key, num_carriers=num_carriers, alpha=a, use_ecc=True)
        st_imgs, _ = engine.embed_text(covers, secret_text)
        p_vals = [compute_psnr(covers[k], st_imgs[k]) for k in range(num_carriers)]
        s_vals = [compute_ssim(covers[k], st_imgs[k]) for k in range(num_carriers)]
        avg_psnr_list.append(np.mean(p_vals))
        avg_ssim_list.append(np.mean(s_vals))
        print(f"  alpha={a:3.1f} -> Avg PSNR: {np.mean(p_vals):.2f} dB | Avg SSIM: {np.mean(s_vals):.4f}")

    g1_path = os.path.join(RESULTS_DIR, "graph1_psnr_ssim_vs_alpha.png")
    plot_psnr_ssim_vs_alpha(alphas, avg_psnr_list, avg_ssim_list, save_path=g1_path)
    plt.close()
    print(f"[+] Saved Graph 1 to '{g1_path}'")

    # -------------------------------------------------------------------------
    # 3. Graph 2a: BER vs. Gaussian Noise (MC-SS vs Single-Image DSSS)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("STEP 3: Evaluating BER vs. Gaussian Noise (Graph 2a)")
    print("-" * 50)
    sigmas = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]
    ber_mcss_noise = []
    ber_dsss_noise = []

    dsss_baseline = SingleImageDSSS(secret_key=secret_key, alpha=2.0)
    dsss_stego, dsss_meta = dsss_baseline.embed_text(covers[0], secret_text)

    for s in sigmas:
        # MC-SS
        noisy_stegos = [apply_gaussian_noise(st, sigma=s) if s > 0 else st for st in stegos]
        _, mcss_bits, _ = mcss.extract_text(noisy_stegos, metadata)
        ber_mcss_noise.append(compute_ber(raw_bits, mcss_bits))

        # DSSS Baseline
        noisy_dsss = apply_gaussian_noise(dsss_stego, sigma=s) if s > 0 else dsss_stego
        _, dsss_bits = dsss_baseline.extract_text(noisy_dsss, dsss_meta)
        ber_dsss_noise.append(compute_ber(raw_bits, dsss_bits))
        print(f"  sigma={s:4.1f} -> MC-SS BER: {ber_mcss_noise[-1]*100:.2f}% | DSSS BER: {ber_dsss_noise[-1]*100:.2f}%")

    g2_path = os.path.join(RESULTS_DIR, "graph2_ber_vs_gaussian_noise.png")
    plot_ber_vs_attack("Gaussian Noise Standard Deviation (sigma)", sigmas, ber_mcss_noise, ber_dsss_noise,
                       "Robustness to AWGN: MC-SS vs. Single-Image DSSS", save_path=g2_path)
    plt.close()
    print(f"[+] Saved Graph 2a to '{g2_path}'")

    # -------------------------------------------------------------------------
    # 4. Graph 2b: BER vs. JPEG Compression (MC-SS vs Single-Image DSSS)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("STEP 4: Evaluating BER vs. JPEG Compression (Graph 2b)")
    print("-" * 50)
    q_factors = [95, 85, 75, 65, 55, 45]
    ber_mcss_jpeg = []
    ber_dsss_jpeg = []

    for qf in q_factors:
        # MC-SS
        jpeg_stegos = [apply_jpeg_compression(st, quality_factor=qf) for st in stegos]
        _, mcss_bits, _ = mcss.extract_text(jpeg_stegos, metadata)
        ber_mcss_jpeg.append(compute_ber(raw_bits, mcss_bits))

        # DSSS Baseline
        jpeg_dsss = apply_jpeg_compression(dsss_stego, quality_factor=qf)
        _, dsss_bits = dsss_baseline.extract_text(jpeg_dsss, dsss_meta)
        ber_dsss_jpeg.append(compute_ber(raw_bits, dsss_bits))
        print(f"  QF={qf:2d} -> MC-SS BER: {ber_mcss_jpeg[-1]*100:.2f}% | DSSS BER: {ber_dsss_jpeg[-1]*100:.2f}%")

    g2b_path = os.path.join(RESULTS_DIR, "graph2b_ber_vs_jpeg.png")
    plot_ber_vs_attack("JPEG Quality Factor (QF)", q_factors, ber_mcss_jpeg, ber_dsss_jpeg,
                       "Robustness to JPEG Compression: MC-SS vs. Single-Image DSSS", save_path=g2b_path)
    plt.close()
    print(f"[+] Saved Graph 2b to '{g2b_path}'")

    # -------------------------------------------------------------------------
    # 5. Graph 3: Carrier Loss Stress Test (Available Carriers)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("STEP 5: Evaluating Carrier Loss / Missing Carriers (Graph 3)")
    print("-" * 50)

    # MC-SS without ECC
    mcss_no_ecc = MCSSStego(secret_key=secret_key, num_carriers=num_carriers, alpha=2.0, use_ecc=False)
    stegos_no_ecc, meta_no_ecc = mcss_no_ecc.embed_text(covers, secret_text)

    # Scenarios: 4 carriers received, 3 received (1 dropped), 2 received (2 dropped), 1 received (3 dropped)
    carriers_avail = [4, 3, 2, 1]
    ber_ecc_list = []
    ber_no_ecc_list = []
    ber_dsss_list = []

    for n_avail in carriers_avail:
        dropped_count = num_carriers - n_avail
        dropped_idxs = list(range(dropped_count))

        # MC-SS with ECC
        st_ecc_dropped = simulate_carrier_drop(stegos, dropped_idxs)
        txt_ecc, bits_ecc, _ = mcss.extract_text(st_ecc_dropped, metadata, dropped_carriers=dropped_idxs)
        ber_ecc = compute_ber(raw_bits, bits_ecc)
        ber_ecc_list.append(ber_ecc)

        # MC-SS without ECC
        st_no_ecc_dropped = simulate_carrier_drop(stegos_no_ecc, dropped_idxs)
        txt_no, bits_no, _ = mcss_no_ecc.extract_text(st_no_ecc_dropped, meta_no_ecc, dropped_carriers=dropped_idxs)
        ber_no_ecc = compute_ber(raw_bits, bits_no)
        ber_no_ecc_list.append(ber_no_ecc)

        # Single-Image DSSS (If carrier dropped or intercepted: 100% loss if dropped)
        if dropped_count > 0:
            ber_dsss = 1.0  # complete loss
        else:
            _, bits_dsss = dsss_baseline.extract_text(dsss_stego, dsss_meta)
            ber_dsss = compute_ber(raw_bits, bits_dsss)
        ber_dsss_list.append(ber_dsss)

        safe_txt_ecc = txt_ecc[:30].encode('ascii', errors='replace').decode('ascii')
        safe_txt_no = txt_no[:30].encode('ascii', errors='replace').decode('ascii')
        print(f"  Available Carriers = {n_avail}/{num_carriers} (Dropped: {dropped_idxs}):")
        print(f"    MC-SS (ECC + Interleaving) BER: {ber_ecc * 100:.2f}% | Text: '{safe_txt_ecc}...'")
        print(f"    MC-SS (No ECC) BER:             {ber_no_ecc * 100:.2f}% | Text: '{safe_txt_no}...'")
        print(f"    Single-Image DSSS BER:           {ber_dsss * 100:.2f}%")

    g3_path = os.path.join(RESULTS_DIR, "graph3_ber_vs_carrier_loss.png")
    plot_ber_vs_carrier_loss(carriers_avail, ber_ecc_list, ber_no_ecc_list, ber_dsss_list, save_path=g3_path)
    plt.close()
    print(f"[+] Saved Graph 3 to '{g3_path}'")

    print("\n" + "=" * 70)
    print(" ALL BENCHMARK PLOTS SUCCESSFULLY GENERATED IN 'results/'")
    print("=" * 70)


if __name__ == "__main__":
    run_all_benchmarks()
