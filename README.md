# Multi-Carrier Spread Spectrum (MC-SS) Image Steganography

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit%20%7C%20Jupyter-red.svg)](https://streamlit.io/)
[![PSNR](https://img.shields.io/badge/PSNR-%3E%2042%20dB-brightgreen.svg)]()
[![BER](https://img.shields.io/badge/Clean%20BER-0.00%25-success.svg)]()
[![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)]()

An advanced, fault-tolerant spatial-domain image steganography framework inspired by **Orthogonal Frequency Division Multiplexing (OFDM)** and **Direct-Sequence Spread Spectrum (DSSS)** telecommunications.

Instead of embedding the entire payload into a single image (a catastrophic single point of failure), **MC-SS** splits the message across $N$ orthogonal image carriers using Linear Feedback Shift Register (LFSR) pseudo-noise (PN) sequences and **Hamming(7,4) error-correcting codes with carrier-level bit interleaving**.

---

## 🏆 Key Architectural Advantages

| Metric / Scenario | Single-Image DSSS (Baseline) | MC-SS (Proposed System) |
| :--- | :--- | :--- |
| **Carrier Loss Fault Tolerance** | **0% Recovery** (100% Catastrophic Loss) | **100% Full Recovery** (0% BER with Hamming ECC) |
| **Visual Imperceptibility** | High embedding noise per pixel | Distributed low-amplitude thermal noise ($\text{PSNR} > 42\text{ dB}$, $\text{SSIM} > 0.99$) |
| **Spreading Processing Gain** | $10 \log_{10}(M / N)$ | $10 \log_{10}(M)$ ($N\times$ higher processing gain per carrier) |
| **Channel Robustness** | Vulnerable to localized bursts/attacks | High resilience to AWGN, JPEG compression, and cropping |
| **Cryptographic Security** | 1 key to intercept | Requires shared seed key; wrong key yields random noise ($\text{BER} \approx 50\%$) |

---

##  Project Directory Structure

```text
DPS/
│
├── dataset/                    # Standard 512×512 benchmark cover images (Lena, Baboon, Peppers, Cameraman, Barbara)
├── results/                    # Generated benchmark evaluation charts & residual heatmaps
├── src/
│   ├── pn_generator.py         # LFSR & Gold-code near-orthogonal PN sequence generator
│   ├── ecc_hamming.py          # Hamming(7,4) ECC encoder/decoder & carrier-level bit interleaver
│   ├── mcss_core.py            # MC-SS embedder, blind correlation extractor & single DSSS baseline
│   ├── attacks.py              # AWGN, JPEG compression, cropping, resizing, and carrier drop simulations
│   └── metrics.py              # PSNR, SSIM, BER computation and publication Matplotlib plots
│
├── app.py                      # Interactive Streamlit Web UI showcase (3 Acts)
├── demo_showcase.ipynb         # Interactive Jupyter Notebook demonstration
├── run_benchmarks.py           # Quantitative evaluation script generating Graphs 1, 2, and 3
├── download_dataset.py         # Automated dataset fetcher and uniform 512x512 preprocessor
├── tests/
│   └── test_pipeline.py        # Comprehensive unit & regression test suite
├── requirements.txt            # Python dependencies
└── README.md                   # System documentation & viva defense guide
```

---

##  Quickstart Guide

### 1. Dedicated Virtual Environment Setup
Ensure you run inside the dedicated Python virtual environment:

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# (Optional) Verify dependencies
pip install -r requirements.txt
```

### 2. Download / Prepare Benchmark Dataset
```powershell
python download_dataset.py
```
This fetches or generates 512×512 uncompressed cover images: **Lena, Baboon, Peppers, Cameraman, and Barbara**.

### 3. Run Quantitative Benchmark Suite
```powershell
python run_benchmarks.py
```
This automatically computes all metrics and outputs publication-quality figures into `results/`:
* `graph1_psnr_ssim_vs_alpha.png`: PSNR & SSIM vs. Embedding Amplitude ($\alpha$).
* `graph2_ber_vs_gaussian_noise.png`: BER vs. AWGN standard deviation ($\sigma$).
* `graph2b_ber_vs_jpeg.png`: BER vs. JPEG Quality Factor ($QF$).
* `graph3_ber_vs_carrier_loss.png`: BER vs. Available Carriers ($N, N-1, N-2$).
* `residual_heatmap.png`: 50×50 magnified thermal-noise residual plot.

### 4. Launch Interactive Streamlit Showcase (Live Mentor Demo)
```powershell
streamlit run app.py
```
Opens the live browser interface structured into **Act 1, Act 2, and Act 3**.

### 5. Launch Jupyter Notebook Showcase
```powershell
jupyter notebook demo_showcase.ipynb
```

### 6. Run Unit Tests
```powershell
python -m unittest discover tests
```

---

##  10-Minute Mentor Showcase Script

### Act 1: Live Interactive Demo (4 Minutes)
1. Open the **Streamlit Web UI** (`streamlit run app.py`).
2. Ask your mentor to enter a custom secret message and seed key.
3. Show the **$N=4$ Cover vs. Stego image pairs** side-by-side:
   - Point to the live **PSNR (> 40 dB)** and **SSIM (> 0.98)** badges to prove visual imperceptibility.
4. Open the **Residual Heatmap**:
   - Point out the zero-mean Gaussian appearance ($I_{\text{stego}} - I_{\text{cover}}$) proving it mimics sensor thermal noise.
5. Click **Extract**:
   - Verify that the message is recovered with **0% Bit Error Rate (BER)**.

### Act 2: Live Attack & Carrier-Loss Stress Test (4 Minutes)
1. Select **Missing Carrier Simulation (Carrier Drop)**:
   - Drop Carrier 1 (Lena).
   - Show that **Single-Image DSSS suffers 100% catastrophic loss**.
   - Show that **MC-SS without ECC recovers the remaining 75%** (graceful degradation).
   - Show that **MC-SS with Hamming(7,4) ECC + Interleaving recovers 100% of the message with 0% BER**!
2. Apply **AWGN** ($\sigma = 12$) or **JPEG Compression** ($QF = 70$):
   - Show that MC-SS maintains lower BER than the single-carrier baseline.
3. Demonstrate **Security Proof**:
   - Enter an incorrect key seed. The adversary extracts random gibberish with $\text{BER} \approx 50\%$.

### Act 3: Quantitative Defense & Viva (2 Minutes)
Review the three pre-computed graphs and answer theoretical questions:
* **OFDM Parallel:** Just as OFDM divides data across orthogonal frequency subcarriers to avoid narrowband fading, MC-SS divides data across orthogonal spatial carriers to survive localized damage and image loss.
* **Processing Gain:** $M \approx 1,500$ pixels per bit yields $\sim 32\text{ dB}$ processing gain, allowing signals with $\alpha = 2$ (PSNR > 42 dB) to be pulled cleanly out of image interference.
* **High-Pass Residual Filtering:** Receiver cancels low-frequency host image interference by subtracting a 3×3 moving average, leaving high-frequency PN chips intact.
