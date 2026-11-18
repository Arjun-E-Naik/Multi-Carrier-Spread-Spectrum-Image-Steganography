"""
tcp_server.py
Standalone TCP/IP Server for Multi-Carrier Spread Spectrum (MC-SS) Image Steganography.
Listens for incoming framed carrier images, logs packet details, and performs blind
correlation extraction using its locally configured secret key.

Usage:
    python tcp_server.py --host 0.0.0.0 --port 5000 --key "QuantumKey#2026"
"""

import argparse
import os
import sys
import time
import cv2

from src.network_tcp import StegoTCPServer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def print_session(session: dict, save_dir: str | None = None) -> None:
    print("\n" + "=" * 72)
    print(f"[SERVER] New TCP/IP Session Completed: {session['client_addr']} -> {session['server_addr']}")
    print("=" * 72)

    for row in session["packet_logs"]:
        print(
            f"  Step {row['Step']:2d} | {row['Packet Type']:18s} | "
            f"{row['Carrier']:22s} | {row['Size (KB)']:7.2f} KB | "
            f"{row['Latency (ms)']:6.2f} ms | {row['Status']}"
        )

    total_kb = session["total_bytes_received"] / 1024.0
    print("-" * 72)
    print(f"  Total Bytes Received : {total_kb:.2f} KB in {session['total_ms']:.2f} ms")
    print(f"  Carriers Received    : {session['received_count']} / {session['metadata']['num_carriers']}")
    print(f"  Dropped Carriers     : {session['dropped_carriers'] if session['dropped_carriers'] else 'None'}")
    print(f"  Server Decode Time   : {session['extract_ms']:.2f} ms")
    print(f"  Server Key Used      : '{session['server_key_used']}'")
    print("-" * 72)
    safe_text = session["extracted_text"].encode("ascii", errors="replace").decode("ascii")
    print(f"  [DECODED MESSAGE]    : {safe_text}")
    print("=" * 72, flush=True)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for idx, img in enumerate(session["received_stegos"]):
            if img is not None:
                out_path = os.path.join(save_dir, f"received_carrier_{idx}.png")
                cv2.imwrite(out_path, img)
        print(f"  [+] Saved received carrier images to '{save_dir}/'")


def main() -> None:
    parser = argparse.ArgumentParser(description="MC-SS Steganography TCP/IP Receiver Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind IP address (use 0.0.0.0 for LAN)")
    parser.add_argument("--port", type=int, default=5000, help="TCP listening port (default: 5000)")
    parser.add_argument("--key", default="QuantumKey#2026", help="Pre-shared secret key seed")
    parser.add_argument("--save-dir", default=None, help="Optional folder to save received stego images")
    args = parser.parse_args()

    server = StegoTCPServer(
        host=args.host,
        port=args.port,
        secret_key=args.key,
        on_session_complete=lambda s: print_session(s, save_dir=args.save_dir),
    )

    bound_host, bound_port = server.start(background=True)
    print("=" * 72)
    print(" MC-SS STEGANOGRAPHY TCP/IP RECEIVER SERVER")
    print("=" * 72)
    print(f"[*] Listening on {bound_host}:{bound_port}")
    print(f"[*] Local Secret Key: '{args.key}' (not transmitted over network)")
    print("[*] Waiting for incoming carrier streams... (Press Ctrl+C to stop)")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[*] Shutting down TCP server...")
    finally:
        server.stop()
        print("[+] Server stopped cleanly.")


if __name__ == "__main__":
    main()
