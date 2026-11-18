"""
tcp_client.py
Standalone TCP/IP Client for Multi-Carrier Spread Spectrum (MC-SS) Image Steganography.
Embeds a secret message into N cover images and transmits them over TCP/IP to a running
StegoTCPServer, with optional live carrier drop or channel attack simulation.

Usage:
    python tcp_client.py --host 127.0.0.1 --port 5000 --key "QuantumKey#2026" \
        --msg "Hello Mentor over TCP/IP!" --drop-carrier 1
"""

import argparse
import os
import sys
import cv2

from download_dataset import download_and_prepare
from src.mcss_core import MCSSStego
from src.metrics import compute_psnr, compute_ssim
from src.network_tcp import StegoTCPClient

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_CARRIERS = ["lena.png", "baboon.png", "peppers.png", "cameraman.png", "barbara.png"]


def load_covers(num_carriers: int):
    download_and_prepare()
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    covers = []
    names = []
    for fname in DEFAULT_CARRIERS[:num_carriers]:
        path = os.path.join(dataset_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Missing cover image: {path}")
        if img.shape != (512, 512):
            img = cv2.resize(img, (512, 512))
        covers.append(img)
        names.append(fname.split(".")[0].capitalize())
    return covers, names


def main() -> None:
    parser = argparse.ArgumentParser(description="MC-SS Steganography TCP/IP Sender Client")
    parser.add_argument("--host", default="127.0.0.1", help="Target server IP address")
    parser.add_argument("--port", type=int, default=5000, help="Target server TCP port")
    parser.add_argument("--key", default="QuantumKey#2026", help="Pre-shared secret key seed")
    parser.add_argument("--msg", default="CONFIDENTIAL: Sent live over TCP/IP using 4-Carrier MC-SS.", help="Secret message")
    parser.add_argument("--carriers", type=int, default=4, choices=[2, 3, 4, 5], help="Number of cover carriers (N)")
    parser.add_argument("--alpha", type=float, default=2.0, help="Embedding amplitude alpha")
    parser.add_argument("--no-ecc", action="store_true", help="Disable Hamming(7,4) ECC")
    parser.add_argument("--drop-carrier", type=int, nargs="*", default=[], help="Carrier indices (0-indexed) to drop in transit")
    parser.add_argument(
        "--attack",
        default="None",
        choices=["None", "AWGN Noise", "JPEG Compression", "Cropping"],
        help="Optional network channel distortion in transit",
    )
    parser.add_argument("--attack-param", type=float, default=0.0, help="Attack intensity (sigma / JPEG QF / crop fraction)")
    args = parser.parse_args()

    covers, carrier_names = load_covers(args.carriers)
    use_ecc = not args.no_ecc

    print("=" * 72)
    print(" MC-SS STEGANOGRAPHY TCP/IP SENDER CLIENT")
    print("=" * 72)
    print(f"[*] Embedding secret message across {args.carriers} carriers (alpha={args.alpha}, ECC={use_ecc})...")
    mcss = MCSSStego(secret_key=args.key, num_carriers=args.carriers, alpha=args.alpha, use_ecc=use_ecc)
    stegos, metadata = mcss.embed_text(covers, args.msg)

    for k in range(args.carriers):
        psnr_val = compute_psnr(covers[k], stegos[k])
        ssim_val = compute_ssim(covers[k], stegos[k])
        print(f"    Carrier {k} ({carrier_names[k]:10s}): PSNR = {psnr_val:.2f} dB | SSIM = {ssim_val:.4f}")

    print(f"\n[*] Connecting to TCP Server at {args.host}:{args.port}...")
    client = StegoTCPClient(host=args.host, port=args.port)
    result = client.send_carriers(
        stegos=stegos,
        metadata=metadata,
        carrier_names=carrier_names,
        drop_carriers=args.drop_carrier,
        channel_attack=args.attack,
        attack_param=args.attack_param,
    )

    print(f"[+] Connected: {result['local_addr']} -> {result['remote_addr']}")
    print("-" * 72)
    for row in result["tx_logs"]:
        print(
            f"  Step {row['Step']:2d} | {row['Packet']:18s} | "
            f"{row['Carrier']:22s} | {row['Size (KB)']:7.2f} KB | "
            f"{row['RTT (ms)']:6.2f} ms | {row['Status']}"
        )
    print("-" * 72)
    total_kb = result["total_bytes_sent"] / 1024.0
    print(f"[+] Total Sent: {total_kb:.2f} KB in {result['total_ms']:.2f} ms")
    print(f"[+] Server ACK: {result['server_ack']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
