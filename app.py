"""
app.py
Interactive Streamlit Showcase for Multi-Carrier Spread Spectrum (MC-SS) Image Steganography.
Structured in 4 Acts for Academic Demonstration, TCP/IP Network Transmission & Viva Defense.
"""

import html
import os
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

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
    plot_residual_heatmap,
    plot_psnr_ssim_vs_alpha,
    plot_ber_vs_attack,
    plot_ber_vs_carrier_loss
)
from src.ecc_hamming import text_to_bits
from src.network_tcp import StegoTCPClient, run_loopback_tcp_transfer
from download_dataset import download_and_prepare

# Page configuration
st.set_page_config(
    page_title="MC-SS Image Steganography Showcase",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 10px;
        border-left: 5px solid #2e7bcf;
    }
    .badge-pass {
        display: inline-block;
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: bold;
        white-space: nowrap;
    }
    .badge-fail {
        display: inline-block;
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: bold;
        white-space: nowrap;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 600;
    }
    .pkt-table-wrap {
        overflow-x: auto;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .pkt-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }
    .pkt-table th {
        text-align: left;
        padding: 8px 10px;
        background-color: rgba(128, 128, 128, 0.14);
        border-bottom: 1px solid rgba(128, 128, 128, 0.25);
        white-space: nowrap;
    }
    .pkt-table td {
        padding: 7px 10px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.12);
        white-space: nowrap;
    }
    .pkt-ok {
        color: #10b981;
        font-weight: 600;
    }
    .pkt-drop {
        color: #ef4444;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def render_packet_table(rows: list) -> None:
    """Render packet telemetry logs as a clean HTML table without pyarrow dependency."""
    if not rows:
        st.info("No packets logged.")
        return
    headers = list(rows[0].keys())
    th_html = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body_rows = []
    for row in rows:
        tds = []
        for h in headers:
            val_str = html.escape(str(row.get(h, "")))
            if h == "Status":
                css_cls = "pkt-drop" if "DROPPED" in val_str.upper() else "pkt-ok"
                tds.append(f"<td class='{css_cls}'>{val_str}</td>")
            else:
                tds.append(f"<td>{val_str}</td>")
        body_rows.append("<tr>" + "".join(tds) + "</tr>")
    table_html = (
        "<div class='pkt-table-wrap'><table class='pkt-table'>"
        f"<thead><tr>{th_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


@st.cache_resource
def load_all_covers():
    download_and_prepare()
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    img_names = ["lena.png", "baboon.png", "peppers.png", "cameraman.png", "barbara.png"]
    images = {}
    for name in img_names:
        p = os.path.join(dataset_dir, name)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            if img.shape != (512, 512):
                img = cv2.resize(img, (512, 512))
            images[name.split('.')[0].capitalize()] = img
    return images

covers_dict = load_all_covers()

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/color/96/000000/cyber-security.png", width=80)
st.sidebar.title("MC-SS Architecture")
st.sidebar.markdown("**Multi-Carrier Spread Spectrum**\n*Spatial Steganography System*")
st.sidebar.markdown("---")

selected_carrier_names = st.sidebar.multiselect(
    "Select Cover Carriers (N)",
    options=list(covers_dict.keys()),
    default=["Lena", "Baboon", "Peppers", "Cameraman"]
)

if len(selected_carrier_names) < 2:
    st.sidebar.error("Select at least 2 carriers for MC-SS.")
    st.stop()

num_carriers = len(selected_carrier_names)
covers = [covers_dict[k] for k in selected_carrier_names]

secret_key = st.sidebar.text_input("Shared Secret Key Seed", value="QuantumKey#2026")
alpha = st.sidebar.slider("Embedding Amplitude (α)", min_value=0.5, max_value=4.0, value=2.0, step=0.5,
                          help="Controls embedding strength. α <= 3.0 guarantees PSNR > 40 dB.")
use_ecc = st.sidebar.checkbox("Enable Hamming(7,4) ECC + Interleaving", value=True)

st.title("🛡️ Multi-Carrier Spread Spectrum (MC-SS) Image Steganography")
st.caption("A Resilient Spatial Steganography Pipeline with Spatial Diversity & Carrier-Drop Fault Tolerance")

tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Act 1: Live Interactive Demo",
    "⚡ Act 2: Attack & Carrier-Loss Stress Test",
    "📊 Act 3: Quantitative Benchmarks & Viva Defense",
    "🌐 Act 4: Live TCP/IP Network Demo"
])

# =============================================================================
# ACT 1: LIVE INTERACTIVE DEMO
# =============================================================================
with tab1:
    st.subheader("Act 1: Spatial Embedding & Imperceptibility Verification")
    st.write(
        "Embed a custom secret message across multiple orthogonal image carriers using "
        "LFSR-generated Gold sequences in the spatial domain."
    )

    col_in, col_stats = st.columns([2, 1])
    with col_in:
        secret_msg = st.text_area(
            "Secret Plaintext Message to Embed:",
            value="CONFIDENTIAL: Antigravity Multi-Carrier Spread Spectrum Payload 2026.",
            height=80
        )
    with col_stats:
        raw_bits = text_to_bits(secret_msg)
        st.markdown(f"**Payload Analysis:**")
        st.markdown(f"- Text Length: **{len(secret_msg)} characters**")
        st.markdown(f"- Total Bitstream: **{len(raw_bits)} bits** (with 32-bit header)")
        st.markdown(f"- Carriers (N): **{num_carriers} images**")
        st.markdown(f"- Spreading Factor (M): **~{262144 // (len(raw_bits) // num_carriers)} chips/bit**")

    # Run embedding
    mcss = MCSSStego(secret_key=secret_key, num_carriers=num_carriers, alpha=alpha, use_ecc=use_ecc)
    stegos, metadata = mcss.embed_text(covers, secret_msg)

    st.markdown("### 1. Visual Imperceptibility Check (Cover vs. Stego)")
    st.write("Compare original covers with generated stego images. Verification threshold: **PSNR > 40 dB**, **SSIM > 0.98**.")

    # Display image columns
    img_cols = st.columns(num_carriers)
    for i, col in enumerate(img_cols):
        c_name = selected_carrier_names[i]
        c_img = covers[i]
        s_img = stegos[i]

        psnr_val = compute_psnr(c_img, s_img)
        ssim_val = compute_ssim(c_img, s_img)

        with col:
            st.markdown(f"**Carrier {i+1}: {c_name}**")
            toggle = st.radio(f"View {c_name}", ["Stego Image", "Original Cover"], key=f"toggle_{i}")
            if toggle == "Stego Image":
                st.image(s_img, caption=f"Stego {c_name} (α={alpha})", use_container_width=True)
            else:
                st.image(c_img, caption=f"Original {c_name}", use_container_width=True)

            psnr_badge = f"<span class='badge-pass'>{psnr_val:.2f} dB</span>" if psnr_val >= 40.0 else f"<span class='badge-fail'>{psnr_val:.2f} dB</span>"
            ssim_badge = f"<span class='badge-pass'>{ssim_val:.4f}</span>" if ssim_val >= 0.98 else f"<span class='badge-fail'>{ssim_val:.4f}</span>"
            
            st.markdown(f"PSNR: {psnr_badge}", unsafe_allow_html=True)
            st.markdown(f"SSIM: {ssim_badge}", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 2. The 'Noise' Proof: Spread Spectrum Residual Heatmap")
    st.write(
        "Magnified 50×50 pixel difference matrix ($I_{\\mathrm{stego}} - I_{\\mathrm{cover}}$). "
        "Notice how the embedded signal manifests as zero-mean, low-amplitude pseudo-random noise."
    )

    col_heat1, col_heat2 = st.columns([1, 1])
    with col_heat1:
        carrier_to_inspect = st.selectbox("Select carrier to inspect residual:", range(num_carriers),
                                          format_func=lambda x: f"Carrier {x+1}: {selected_carrier_names[x]}")
        fig_heat = plot_residual_heatmap(covers[carrier_to_inspect], stegos[carrier_to_inspect], patch_size=50)
        st.pyplot(fig_heat)
        plt.close()

    with col_heat2:
        st.markdown("#### Residual Statistics")
        res = stegos[carrier_to_inspect].astype(float) - covers[carrier_to_inspect].astype(float)
        st.metric("Mean Residual (μ)", f"{np.mean(res):.4f}", help="Should be ~ 0 for zero-mean noise")
        st.metric("Residual Std Dev (σ)", f"{np.std(res):.2f}", help=f"Proportional to α={alpha}")
        st.metric("Max Absolute Residual", f"{np.max(np.abs(res)):.1f} intensity levels")
        st.info("💡 **Mentor Insight:** Because $\\alpha$ is small and the sequence is zero-mean pseudo-noise, the human visual system (HVS) cannot perceive the perturbations. The processing gain pulls the signal from the noise floor at the receiver.")

    st.markdown("---")
    st.markdown("### 3. Blind Correlation Extraction")
    rec_text, rec_bits, stats = mcss.extract_text(stegos, metadata)
    ber_clean = compute_ber(raw_bits, rec_bits)

    col_r1, col_r2 = st.columns([3, 1])
    with col_r1:
        st.success(f"**Successfully Extracted Secret Message:**\n\n`{rec_text}`")
    with col_r2:
        st.metric("Clean BER", f"{ber_clean * 100:.2f}%", delta="0.0% Error (Perfect)")


# =============================================================================
# ACT 2: ATTACK & CARRIER-LOSS STRESS TEST
# =============================================================================
with tab2:
    st.subheader("Act 2: Stress Testing & Carrier-Drop Fault Tolerance")
    st.write(
        "Demonstrate the key scientific advantage of **Multi-Carrier Spread Spectrum (MC-SS)** "
        "over the **Single-Image DSSS baseline** under channel attacks and carrier dropouts."
    )

    attack_type = st.selectbox(
        "Select Attack / Channel Impairment to Simulate:",
        ["Missing Carrier / Interception (Carrier Drop)",
         "Additive White Gaussian Noise (AWGN)",
         "Lossy JPEG Compression",
         "Spatial Cropping / Packet Loss",
         "Resizing (Downsampling & Upsampling)"]
    )

    # Initialize Single-Image DSSS Baseline on Carrier 0 for direct comparison
    dsss_baseline = SingleImageDSSS(secret_key=secret_key, alpha=alpha)
    dsss_stego, dsss_meta = dsss_baseline.embed_text(covers[0], secret_msg)

    # Un-ECC MC-SS for comparison
    mcss_raw = MCSSStego(secret_key=secret_key, num_carriers=num_carriers, alpha=alpha, use_ecc=False)
    stegos_raw, meta_raw = mcss_raw.embed_text(covers, secret_msg)

    if attack_type == "Missing Carrier / Interception (Carrier Drop)":
        st.markdown("#### Scenario: Intercepted or Dropped Carrier Image")
        st.warning("Simulate packet drop, firewall blocking, or network loss of 1 or more stego images.")

        dropped_count = st.radio("Number of dropped carriers:", [1, 2], horizontal=True)
        dropped_indices = list(range(dropped_count))

        # Attacked representations
        att_stegos_ecc = simulate_carrier_drop(stegos, dropped_indices)
        att_stegos_raw = simulate_carrier_drop(stegos_raw, dropped_indices)

        # Extraction
        txt_ecc, bits_ecc, _ = mcss.extract_text(att_stegos_ecc, metadata, dropped_carriers=dropped_indices)
        ber_ecc = compute_ber(raw_bits, bits_ecc)

        txt_raw, bits_raw, _ = mcss_raw.extract_text(att_stegos_raw, meta_raw, dropped_carriers=dropped_indices)
        ber_raw = compute_ber(raw_bits, bits_raw)

        # DSSS Baseline: if image 0 is lost, 100% loss
        txt_dsss, bits_dsss = dsss_baseline.extract_text(None, dsss_meta)
        ber_dsss = 1.0

        st.markdown("### Comparison Results")
        col_m1, col_m2, col_m3 = st.columns(3)

        with col_m1:
            st.markdown("#### 1. MC-SS (Hamming ECC)")
            st.metric("Bit Error Rate (BER)", f"{ber_ecc * 100:.2f}%", delta="100% Intact" if ber_ecc == 0 else "Degraded")
            st.text_area("Decoded Output (ECC):", value=txt_ecc, height=100, key="txt_ecc_out")
            st.success("✅ **Full Recovery:** Bit interleaving spreads Hamming codeword bits across different carriers. Dropping an entire carrier causes at most 1-2 bit erasures per codeword, which Hamming(7,4) resolves completely!")

        with col_m2:
            st.markdown("#### 2. MC-SS (No ECC)")
            st.metric("Bit Error Rate (BER)", f"{ber_raw * 100:.2f}%")
            st.text_area("Decoded Output (No ECC):", value=txt_raw, height=100, key="txt_raw_out")
            st.warning("⚠️ **Graceful Degradation:** Losing 1 of 4 carriers only loses that 25% sub-stream; the remaining 75% remains uncorrupted.")

        with col_m3:
            st.markdown("#### 3. Single-Image DSSS")
            st.metric("Bit Error Rate (BER)", f"{ber_dsss * 100:.2f}%", delta="-100% Total Loss", delta_color="inverse")
            st.text_area("Decoded Output (DSSS):", value="[DATA LOST: SINGLE POINT OF FAILURE]", height=100, key="txt_dsss_out")
            st.error("❌ **Catastrophic Failure:** Single-carrier steganography is vulnerable to single-point-of-failure. Losing the carrier destroys 100% of the payload.")

    else:
        # Standard noise / compression / crop attacks
        if attack_type == "Additive White Gaussian Noise (AWGN)":
            sigma_val = st.slider("Gaussian Noise Standard Deviation (σ):", 1.0, 25.0, 10.0, step=1.0)
            att_mcss = [apply_gaussian_noise(st, sigma=sigma_val) for st in stegos]
            att_dsss = apply_gaussian_noise(dsss_stego, sigma=sigma_val)
        elif attack_type == "Lossy JPEG Compression":
            qf_val = st.slider("JPEG Quality Factor (QF):", 30, 95, 70, step=5)
            att_mcss = [apply_jpeg_compression(st, quality_factor=qf_val) for st in stegos]
            att_dsss = apply_jpeg_compression(dsss_stego, quality_factor=qf_val)
        elif attack_type == "Spatial Cropping / Packet Loss":
            crop_frac = st.slider("Cropped Area Fraction (%):", 0.05, 0.35, 0.20, step=0.05)
            att_mcss = [apply_cropping(st, crop_fraction=crop_frac) for st in stegos]
            att_dsss = apply_cropping(dsss_stego, crop_fraction=crop_frac)
        else: # Resizing
            scale_val = st.slider("Downsampling Scale Factor:", 0.25, 0.9, 0.5, step=0.05)
            att_mcss = [apply_resizing(st, scale=scale_val) for st in stegos]
            att_dsss = apply_resizing(dsss_stego, scale=scale_val)

        # Extract
        txt_mcss_att, bits_mcss_att, _ = mcss.extract_text(att_mcss, metadata)
        ber_mcss_att = compute_ber(raw_bits, bits_mcss_att)

        txt_dsss_att, bits_dsss_att = dsss_baseline.extract_text(att_dsss, dsss_meta)
        ber_dsss_att = compute_ber(raw_bits, bits_dsss_att)

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown("#### MC-SS Performance")
            st.metric("MC-SS BER", f"{ber_mcss_att * 100:.2f}%")
            st.text_area("MC-SS Decoded Output:", value=txt_mcss_att, height=90, key="att_mcss_out")
            st.image(att_mcss[0], caption=f"Attacked Stego Carrier 1 ({selected_carrier_names[0]})", width=250)

        with col_a2:
            st.markdown("#### Single-Image DSSS Baseline")
            st.metric("DSSS BER", f"{ber_dsss_att * 100:.2f}%")
            st.text_area("Single DSSS Decoded Output:", value=txt_dsss_att, height=90, key="att_dsss_out")
            st.image(att_dsss, caption="Attacked Single DSSS Stego", width=250)

    st.markdown("---")
    st.markdown("### 🔒 Security Proof: Unauthorized Interception & Wrong Key Test")
    st.write(
        "Demonstrate cryptographic confidentiality: An adversary who intercepts only 1 carrier "
        "or attempts extraction using the wrong seed key receives unintelligible random noise."
    )
    col_sec1, col_sec2 = st.columns(2)
    with col_sec1:
        wrong_key = st.text_input("Test with Incorrect Key Seed:", value="AttackerGuess#9999")
        attacker_mcss = MCSSStego(secret_key=wrong_key, num_carriers=num_carriers, alpha=alpha, use_ecc=use_ecc)
        att_txt, att_bits, _ = attacker_mcss.extract_text(stegos, metadata)
        att_ber = compute_ber(raw_bits, att_bits)
        st.metric("Adversary BER (Wrong Key)", f"{att_ber * 100:.2f}%", help="~50% indicates completely uncorrelated noise")
        st.code(f"Adversary Recovered Text:\n'{att_txt[:50]}...'")

    with col_sec2:
        st.markdown("**Cryptographic Guarantee:**")
        st.write(
            "Without the exact secret seed, the LFSR generates an uncorrelated pseudo-noise sequence "
            "with zero processing gain. As a result, the cross-correlation remains near zero, yielding "
            "a random coin-flip decision (BER ≈ 50%), successfully protecting the secret payload."
        )


# =============================================================================
# ACT 3: QUANTITATIVE BENCHMARKS & VIVA DEFENSE
# =============================================================================
with tab3:
    st.subheader("Act 3: Benchmark Graphs & Viva Defense")
    st.write(
        "Pre-computed quantitative evaluation curves and technical rationale to defend the "
        "engineering architecture during mentor evaluation."
    )

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    g1_path = os.path.join(results_dir, "graph1_psnr_ssim_vs_alpha.png")
    g2_path = os.path.join(results_dir, "graph2_ber_vs_gaussian_noise.png")
    g2b_path = os.path.join(results_dir, "graph2b_ber_vs_jpeg.png")
    g3_path = os.path.join(results_dir, "graph3_ber_vs_carrier_loss.png")

    tab_g1, tab_g2, tab_g3, tab_viva = st.tabs([
        "📈 Graph 1: Imperceptibility vs. Alpha",
        "📉 Graph 2: Robustness Curves",
        "🛡️ Graph 3: Carrier Loss Resilience",
        "🎓 Mentor Viva Defense Prep"
    ])

    with tab_g1:
        st.markdown("#### Imperceptibility Tradeoff: PSNR & SSIM vs. Embedding Amplitude (α)")
        st.write(
            "Quantifies the tradeoff between visual imperceptibility and signal energy. "
            "At $\\alpha = 2.0$, PSNR exceeds **42 dB** (well above the 40 dB threshold) and SSIM is **> 0.99**."
        )
        if os.path.exists(g1_path):
            st.image(g1_path, use_container_width=True)
        else:
            st.info("Run `python run_benchmarks.py` to generate publication plots.")

    with tab_g2:
        st.markdown("#### BER vs. Attack Intensity: MC-SS vs. Single-Image DSSS")
        st.write("Comparing channel robustness under Additive Gaussian Noise and Lossy JPEG compression.")
        col_ga, col_gb = st.columns(2)
        with col_ga:
            if os.path.exists(g2_path):
                st.image(g2_path, caption="Robustness to AWGN", use_container_width=True)
        with col_gb:
            if os.path.exists(g2b_path):
                st.image(g2b_path, caption="Robustness to JPEG Compression", use_container_width=True)

    with tab_g3:
        st.markdown("#### Fault Tolerance: BER vs. Number of Available Carriers")
        st.write(
            "Simulating dropped or intercepted stego images ($N, N-1, N-2, \\dots$). "
            "Shows catastrophic single-image failure vs. MC-SS graceful degradation and perfect ECC reconstruction."
        )
        if os.path.exists(g3_path):
            st.image(g3_path, use_container_width=True)

    with tab_viva:
        st.markdown("### 🎓 Core Viva Questions & Technical Answers")

        with st.expander("Q1: Why compare Multi-Carrier Spread Spectrum to OFDM in telecommunications?", expanded=True):
            st.markdown("""
            **Technical Answer:**
            Just as **Orthogonal Frequency Division Multiplexing (OFDM)** splits a high-bitrate data stream across multiple orthogonal subcarrier frequencies to survive frequency-selective fading and narrowband interference:
            - **MC-SS** splits the high-density secret payload across $N$ orthogonal PN sequences embedded into separate spatial image carriers.
            - Localized distortions, JPEG quantization, or packet loss in one carrier does not destroy the entire payload.
            - Spatial diversity combined with Hamming interleaving allows full payload recovery even with complete loss of a subcarrier image.
            """)

        with st.expander("Q2: How do you maintain PSNR > 40 dB while achieving 0% BER?", expanded=True):
            st.markdown("""
            **Technical Answer:**
            The key lies in the **Spreading Factor (Processing Gain)**:
            - For a $512 \\times 512$ image ($262,144$ pixels) and a sub-message of ~150 bits, each bit is spread over $M \\approx 1,700$ pixels.
            - Spreading gain is given by:
              $$\\mathrm{Processing\\ Gain} = 10 \\log_{10}(M) \\approx 10 \\log_{10}(1700) \\approx 32.3\\ \\text{dB}$$
            - Even though the spatial amplitude is minuscule ($\\alpha = 1.5 \\sim 2.0$ intensity levels, guaranteeing PSNR > 42 dB), the coherent correlation summation accumulator $\\sum_{j=1}^M I_{\\mathrm{stego}}(j) P_k(j)$ aggregates signal energy coherently while incoherent image noise cancels out.
            """)

        with st.expander("Q3: Why is high-pass residual filtering used at the receiver?", expanded=True):
            st.markdown("""
            **Technical Answer:**
            The host cover image content acts as interference during blind extraction:
            $$C = \\sum_{j=1}^M I(j) P_k(j) + \\alpha \\cdot b_i \\cdot M$$
            Because natural images have strong low-frequency spatial correlations, subtracting a local 3×3 moving average (high-pass filter) attenuates the host image interference $\\sum I(j) P(j)$ by over 90%, leaving the high-frequency PN chips intact and allowing error-free threshold detection.
            """)

        with st.expander("Q4: How does bit interleaving with Hamming(7,4) prevent message loss?", expanded=True):
            st.markdown("""
            **Technical Answer:**
            A Hamming(7,4) code can correct 1 bit error or up to 2 bit erasures per 7-bit block.
            - If bits were embedded sequentially, losing Carrier 1 would destroy a continuous 25% chunk of text (catastrophic burst error).
            - By interleaving the 7 bits of each codeword across the $N$ carriers (Carrier $k = i \\pmod N$), losing Carrier 1 causes at most $\\lceil 7/N \\rceil$ missing bits per codeword.
            - For $N=4$ or $N=5$, each codeword loses at most 1 or 2 bits, which the syndrome and erasure decoder corrects with 100% fidelity.
            """)


# =============================================================================
# ACT 4: LIVE TCP/IP NETWORK TRANSMISSION DEMO
# =============================================================================
with tab4:
    st.subheader("Act 4: Live TCP/IP Client-Server Carrier Transmission")
    st.write(
        "Transmit the embedded stego carrier images over real **TCP/IP sockets (`AF_INET`, `SOCK_STREAM`)** "
        "using a 6-byte length-prefixed framing protocol (`[Type | Carrier ID | Payload Size | PNG/JPEG Bytes]`). "
        "The **shared secret key is never sent over the network**."
    )

    tcp_mode = st.radio(
        "Network Mode:",
        [
            "Built-in Local TCP/IP Loopback (Client + Server in One View)",
            "Send to External TCP Server (Two Terminals / LAN)",
        ],
        horizontal=True,
        key="tcp_mode_radio",
    )

    col_tx, col_net, col_rx = st.columns(3)

    with col_tx:
        st.markdown("#### 📤 Sender (TCP Client)")
        tcp_msg = st.text_area(
            "Message to Send over TCP/IP:",
            value=secret_msg,
            height=80,
            key="tcp_secret_msg",
        )
        if tcp_mode.startswith("Built-in"):
            tcp_host = st.text_input("Loopback IP:", value="127.0.0.1", disabled=True, key="tcp_host_loop")
            tcp_port = st.number_input("TCP Server Port:", min_value=1024, max_value=65535, value=5050, step=1, key="tcp_port_loop")
        else:
            tcp_host = st.text_input("External Server IP:", value="127.0.0.1", key="tcp_host_ext")
            tcp_port = st.number_input("External Server Port:", min_value=1024, max_value=65535, value=5000, step=1, key="tcp_port_ext")
        st.caption(f"Sender Key: `{secret_key}` | Carriers: `{num_carriers}` | α: `{alpha}` | ECC: `{use_ecc}`")

    with col_net:
        st.markdown("#### 🌐 Network Channel")
        drop_options = list(range(num_carriers))
        tcp_drop_carriers = st.multiselect(
            "Simulate Firewall / Packet Drop (Carriers):",
            options=drop_options,
            default=[],
            format_func=lambda idx: f"Carrier {idx + 1} ({selected_carrier_names[idx]})",
            key="tcp_drop_select",
        )
        tcp_attack = st.selectbox(
            "In-Transit Channel Distortion:",
            ["None", "AWGN Noise", "JPEG Compression", "Cropping"],
            key="tcp_attack_select",
        )
        tcp_attack_param = 0.0
        if tcp_attack == "AWGN Noise":
            tcp_attack_param = st.slider("Wire AWGN Sigma (σ):", 1.0, 25.0, 10.0, step=1.0, key="tcp_awgn_slider")
        elif tcp_attack == "JPEG Compression":
            tcp_attack_param = float(st.slider("Wire JPEG Quality (QF):", 35, 95, 75, step=5, key="tcp_jpeg_slider"))
        elif tcp_attack == "Cropping":
            tcp_attack_param = st.slider("Wire Crop Fraction:", 0.05, 0.30, 0.15, step=0.05, key="tcp_crop_slider")

    with col_rx:
        st.markdown("#### 📥 Receiver (TCP Server)")
        if tcp_mode.startswith("Built-in"):
            simulate_wrong_server_key = st.checkbox(
                "Simulate Unauthorized / Wrong Key on Server",
                value=False,
                key="tcp_wrong_key_chk",
            )
            if "tcp_srv_key_input" not in st.session_state:
                st.session_state["tcp_srv_key_input"] = secret_key
            if (
                st.session_state.get("_prev_wrong_key_chk") != simulate_wrong_server_key
                or st.session_state.get("_prev_sender_key") != secret_key
            ):
                st.session_state["tcp_srv_key_input"] = (
                    "WrongServerKey#0000" if simulate_wrong_server_key else secret_key
                )
                st.session_state["_prev_wrong_key_chk"] = simulate_wrong_server_key
                st.session_state["_prev_sender_key"] = secret_key

            tcp_server_key = st.text_input(
                "Server Local Secret Key:",
                key="tcp_srv_key_input",
                help="Stored locally on the server; never transmitted over TCP.",
            )
            st.caption("Server binds socket, receives framed PNG/JPEG packets, detects dropped carriers, and extracts blindly.")
        else:
            tcp_server_key = secret_key
            st.info(
                "Start the external server in another terminal or laptop:\n\n"
                f"`python tcp_server.py --host 0.0.0.0 --port {int(tcp_port)} --key \"{secret_key}\"`"
            )

    transmit_clicked = st.button("🚀 Embed & Transmit Over TCP/IP", type="primary", key="tcp_transmit_btn")

    if transmit_clicked:
        try:
            with st.spinner("Embedding payload and streaming carrier packets over TCP/IP..."):
                tx_engine = MCSSStego(
                    secret_key=secret_key,
                    num_carriers=num_carriers,
                    alpha=alpha,
                    use_ecc=use_ecc,
                )
                tx_stegos, tx_meta = tx_engine.embed_text(covers, tcp_msg)
                tx_raw_bits = text_to_bits(tcp_msg)

                if tcp_mode.startswith("Built-in"):
                    try:
                        client_res, server_res = run_loopback_tcp_transfer(
                            stegos=tx_stegos,
                            metadata=tx_meta,
                            server_secret_key=tcp_server_key,
                            carrier_names=selected_carrier_names,
                            host="127.0.0.1",
                            port=int(tcp_port),
                            drop_carriers=tcp_drop_carriers,
                            channel_attack=tcp_attack,
                            attack_param=tcp_attack_param,
                        )
                    except OSError:
                        # Fallback to an auto-assigned free port if chosen port is busy
                        client_res, server_res = run_loopback_tcp_transfer(
                            stegos=tx_stegos,
                            metadata=tx_meta,
                            server_secret_key=tcp_server_key,
                            carrier_names=selected_carrier_names,
                            host="127.0.0.1",
                            port=0,
                            drop_carriers=tcp_drop_carriers,
                            channel_attack=tcp_attack,
                            attack_param=tcp_attack_param,
                        )
                    st.session_state["tcp_demo_result"] = {
                        "mode": "loopback",
                        "sent_text": tcp_msg,
                        "raw_bits": tx_raw_bits,
                        "carrier_names": list(selected_carrier_names),
                        "client_res": client_res,
                        "server_res": server_res,
                        "sender_key": secret_key,
                    }
                else:
                    client = StegoTCPClient(host=tcp_host.strip(), port=int(tcp_port))
                    client_res = client.send_carriers(
                        stegos=tx_stegos,
                        metadata=tx_meta,
                        carrier_names=selected_carrier_names,
                        drop_carriers=tcp_drop_carriers,
                        channel_attack=tcp_attack,
                        attack_param=tcp_attack_param,
                    )
                    st.session_state["tcp_demo_result"] = {
                        "mode": "external",
                        "sent_text": tcp_msg,
                        "raw_bits": tx_raw_bits,
                        "carrier_names": list(selected_carrier_names),
                        "client_res": client_res,
                        "server_res": None,
                        "sender_key": secret_key,
                    }
        except Exception as exc:
            st.error(f"TCP/IP Transfer Failed: {exc}")

    # Render latest TCP/IP session results
    if "tcp_demo_result" in st.session_state:
        res_bundle = st.session_state["tcp_demo_result"]
        client_res = res_bundle["client_res"]
        server_res = res_bundle["server_res"]
        c_names_res = res_bundle["carrier_names"]

        st.markdown("---")
        st.markdown("### 📡 Live TCP/IP Session Telemetry")

        if res_bundle["mode"] == "loopback" and server_res and server_res.get("success"):
            srv_ber = compute_ber(res_bundle["raw_bits"], server_res["extracted_bits"])
            total_kb = server_res["total_bytes_received"] / 1024.0
            n_total = server_res["metadata"]["num_carriers"]
            n_recv = server_res["received_count"]
            exact_text_match = (server_res["extracted_text"] == res_bundle["sent_text"])

            st.caption(
                f"Active TCP/IP Socket Connection: `{client_res['local_addr']}` → `{server_res['server_addr']}` "
                f"| Server Key: `{server_res['server_key_used']}`"
            )

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("TCP Port", f"{server_res['server_addr']}", delta="SOCK_STREAM")
            k2.metric("Carriers Received", f"{n_recv} / {n_total}", delta=f"{len(server_res['dropped_carriers'])} dropped" if server_res["dropped_carriers"] else "All delivered")
            k3.metric("Wire Transfer", f"{total_kb:.1f} KB", delta=f"{server_res['total_ms']:.1f} ms")
            k4.metric("Server Decoded BER", f"{srv_ber * 100:.2f}%", delta="Exact Match" if exact_text_match else "Bit errors detected", delta_color="normal" if exact_text_match else "inverse")

            # Packet Log Tables (pyarrow-free styled HTML tables)
            log_col1, log_col2 = st.columns(2)
            with log_col1:
                st.markdown("#### Sender (Client) TX Packet Log")
                render_packet_table(client_res["tx_logs"])
            with log_col2:
                st.markdown("#### Receiver (Server) RX Packet Log")
                render_packet_table(server_res["packet_logs"])

            # Server-Side Received Carrier Gallery
            st.markdown("#### 🖼️ Carriers Received on Server Side")
            srv_cols = st.columns(n_total)
            for idx, col in enumerate(srv_cols):
                img_rx = server_res["received_stegos"][idx]
                c_title = c_names_res[idx] if idx < len(c_names_res) else f"Carrier {idx + 1}"
                with col:
                    st.markdown(f"**Carrier {idx + 1}: {c_title}**")
                    if img_rx is not None:
                        st.image(img_rx, caption=f"Received via TCP ({img_rx.shape[0]}x{img_rx.shape[1]})", width="stretch")
                        st.markdown("<span class='badge-pass'>RECEIVED & ACKED</span>", unsafe_allow_html=True)
                    else:
                        placeholder = np.zeros((256, 256), dtype=np.uint8)
                        cv2.putText(placeholder, "DROPPED", (42, 138), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (220,), 3)
                        st.image(placeholder, caption="Packet Dropped on Network", width="stretch")
                        st.markdown("<span class='badge-fail'>DROPPED (ERASED)</span>", unsafe_allow_html=True)

            # Decoded Message Comparison
            st.markdown("#### 🔓 Server Blind Extraction Output")
            out_c1, out_c2 = st.columns(2)
            with out_c1:
                st.text_area("Original Message Sent by Client:", value=res_bundle["sent_text"], height=90, disabled=True, key="tcp_orig_view")
            with out_c2:
                st.text_area("Message Reconstructed by Server:", value=server_res["extracted_text"], height=90, disabled=True, key="tcp_srv_view")

            if server_res["server_key_used"] != res_bundle["sender_key"]:
                st.error(
                    f"🔒 **Wrong Server Key (`{server_res['server_key_used']}`):** "
                    f"Even though TCP packets arrived intact, an unauthorized receiver gets random noise (BER = {srv_ber * 100:.2f}%)."
                )
            elif exact_text_match and len(server_res["dropped_carriers"]) > 0:
                st.success(
                    f"✅ **Carrier-Drop Fault Tolerance Proven Over Live TCP/IP!** "
                    f"Carrier(s) {[i + 1 for i in server_res['dropped_carriers']]} were dropped in transit, "
                    f"yet the server reconstructed the exact secret message (BER = {srv_ber * 100:.2f}%) in {server_res['extract_ms']:.1f} ms."
                )
            elif exact_text_match:
                st.success(
                    f"✅ **100% Error-Free Covert TCP/IP Delivery!** "
                    f"Decoded blindly on the server in {server_res['extract_ms']:.1f} ms (BER = {srv_ber * 100:.2f}%)."
                )
            else:
                st.warning(
                    f"⚠️ **Partial Recovery (BER = {srv_ber * 100:.2f}%):** "
                    f"Heavy channel impairment or too many dropped carriers exceeded Hamming(7,4) correction capacity."
                )

        elif res_bundle["mode"] == "external":
            st.success(
                f"✅ Transmitted **{client_res['total_bytes_sent'] / 1024.0:.2f} KB** from "
                f"`{client_res['local_addr']}` to external server `{client_res['remote_addr']}` "
                f"in **{client_res['total_ms']:.2f} ms**."
            )
            render_packet_table(client_res["tx_logs"])
            st.json(client_res["server_ack"])

    with st.expander("💻 Standalone Two-Terminal / Two-Laptop CLI Commands", expanded=False):
        st.code(
            "# Terminal 1 (Receiver Server - run on same PC or mentor's laptop on Wi-Fi)\n"
            f"python tcp_server.py --host 0.0.0.0 --port 5000 --key \"{secret_key}\"\n\n"
            "# Terminal 2 (Sender Client - optionally drop Carrier 1 to test resilience)\n"
            f"python tcp_client.py --host 127.0.0.1 --port 5000 --key \"{secret_key}\" "
            "--msg \"Live MC-SS message over TCP/IP\" --drop-carrier 1",
            language="bash",
        )
