"""
src/network_tcp.py
Real TCP/IP socket client-server implementation for Multi-Carrier Spread Spectrum (MC-SS)
Image Steganography.

Wire Framing Protocol (6-byte Big-Endian Header + Variable Payload):
    +---------------+---------------+--------------------+---------------------+
    | Packet Type   | Carrier Index | Payload Length     | Payload Bytes       |
    | (1 byte, u8)  | (1 byte, u8)  | (4 bytes, u32)     | (N bytes)           |
    +---------------+---------------+--------------------+---------------------+

Packet Types:
    0x01 (PKT_METADATA) : JSON session metadata (shared secret key is NEVER sent)
    0x02 (PKT_CARRIER)  : Encoded stego carrier image bytes (PNG or JPEG)
    0x03 (PKT_END)      : End-of-transmission marker
    0x04 (PKT_ACK)      : Server acknowledgment message (JSON)
"""

import hashlib
import json
import socket
import struct
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.attacks import apply_cropping, apply_gaussian_noise
from src.mcss_core import MCSSStego

# Protocol constants
HEADER_FMT = "!BBI"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

PKT_METADATA = 0x01
PKT_CARRIER = 0x02
PKT_END = 0x03
PKT_ACK = 0x04

NO_CARRIER = 0xFF


def recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    """Read exactly `num_bytes` from a TCP stream socket."""
    chunks = []
    remaining = num_bytes
    while remaining > 0:
        chunk = sock.recv(min(remaining, 65536))
        if not chunk:
            raise ConnectionError(f"TCP connection closed with {remaining} bytes remaining.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_packet(sock: socket.socket, packet_type: int, carrier_idx: int, payload: bytes) -> int:
    """Frame and send a single packet over TCP. Returns total bytes transmitted."""
    header = struct.pack(HEADER_FMT, packet_type, carrier_idx, len(payload))
    sock.sendall(header + payload)
    return len(header) + len(payload)


def recv_packet(sock: socket.socket) -> Tuple[int, int, bytes]:
    """Receive and unframe a single packet from a TCP socket."""
    header = recv_exact(sock, HEADER_SIZE)
    packet_type, carrier_idx, payload_len = struct.unpack(HEADER_FMT, header)
    payload = recv_exact(sock, payload_len) if payload_len > 0 else b""
    return packet_type, carrier_idx, payload


def encode_image_for_wire(image: np.ndarray, wire_format: str = "PNG", jpeg_quality: int = 75) -> bytes:
    """Encode an image array into bytes for TCP transmission."""
    if wire_format.upper() == "JPEG":
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    else:
        ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image as {wire_format}.")
    return buf.tobytes()


def decode_image_from_wire(payload: bytes) -> np.ndarray:
    """Decode received image bytes back into a 2D grayscale numpy array."""
    arr = np.frombuffer(payload, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to decode received image payload.")
    return img


class StegoTCPServer:
    """
    TCP/IP Server that receives framed MC-SS carrier images, tracks missing/dropped
    carriers, logs packet telemetry, and performs blind steganographic extraction
    using its locally configured secret key.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5000,
        secret_key: str = "QuantumKey#2026",
        on_session_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.host = host
        self.port = port
        self.secret_key = secret_key
        self.on_session_complete = on_session_complete

        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()

        self.sessions: List[Dict[str, Any]] = []
        self.last_session: Optional[Dict[str, Any]] = None
        self.bound_address: Optional[Tuple[str, int]] = None

    def start(self, background: bool = True) -> Tuple[str, int]:
        """Bind socket and start accepting TCP connections."""
        self._stop_event.clear()
        self._ready_event.clear()

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(5)
        self._sock.settimeout(0.3)
        self.bound_address = self._sock.getsockname()
        self.port = self.bound_address[1]
        self._ready_event.set()

        if background:
            self._thread = threading.Thread(target=self._serve_loop, daemon=True)
            self._thread.start()
        else:
            self._serve_loop()

        return self.bound_address

    def stop(self) -> None:
        """Stop the server loop and close the listening socket."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _serve_loop(self) -> None:
        assert self._sock is not None
        while not self._stop_event.is_set():
            try:
                conn, client_addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                self._handle_client(conn, client_addr)
            except Exception as exc:
                self.last_session = {
                    "success": False,
                    "error": str(exc),
                    "client_addr": f"{client_addr[0]}:{client_addr[1]}",
                }
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    def _handle_client(self, conn: socket.socket, client_addr: Tuple[str, int]) -> None:
        conn.settimeout(10.0)
        session_start = time.perf_counter()
        server_addr_str = (
            f"{self.bound_address[0]}:{self.bound_address[1]}"
            if self.bound_address
            else f"{self.host}:{self.port}"
        )
        client_addr_str = f"{client_addr[0]}:{client_addr[1]}"

        packet_logs: List[Dict[str, Any]] = []
        metadata: Optional[Dict[str, Any]] = None
        received_map: Dict[int, np.ndarray] = {}
        total_bytes_received = 0

        while True:
            pkt_t0 = time.perf_counter()
            pkt_type, carrier_idx, payload = recv_packet(conn)
            pkt_ms = (time.perf_counter() - pkt_t0) * 1000.0
            frame_bytes = HEADER_SIZE + len(payload)
            total_bytes_received += frame_bytes

            if pkt_type == PKT_METADATA:
                metadata = json.loads(payload.decode("utf-8"))
                packet_logs.append({
                    "Step": len(packet_logs) + 1,
                    "Direction": f"{client_addr_str} -> {server_addr_str}",
                    "Packet Type": "METADATA (0x01)",
                    "Carrier": "Session Header",
                    "Size (KB)": round(frame_bytes / 1024.0, 2),
                    "Latency (ms)": round(pkt_ms, 2),
                    "SHA-256 (12H)": hashlib.sha256(payload).hexdigest()[:12],
                    "Status": "ACKED",
                })
                ack_payload = json.dumps({"status": "READY"}).encode("utf-8")
                send_packet(conn, PKT_ACK, NO_CARRIER, ack_payload)

            elif pkt_type == PKT_CARRIER:
                if metadata is None:
                    raise RuntimeError("Received carrier packet before session metadata.")

                img = decode_image_from_wire(payload)
                received_map[carrier_idx] = img

                carrier_names = metadata.get("carrier_names", [])
                c_label = (
                    f"Carrier {carrier_idx + 1} ({carrier_names[carrier_idx]})"
                    if carrier_idx < len(carrier_names)
                    else f"Carrier {carrier_idx + 1}"
                )
                fmt = "JPEG" if payload[:2] == b"\xff\xd8" else "PNG"

                packet_logs.append({
                    "Step": len(packet_logs) + 1,
                    "Direction": f"{client_addr_str} -> {server_addr_str}",
                    "Packet Type": f"CARRIER {fmt} (0x02)",
                    "Carrier": c_label,
                    "Size (KB)": round(frame_bytes / 1024.0, 2),
                    "Latency (ms)": round(pkt_ms, 2),
                    "SHA-256 (12H)": hashlib.sha256(payload).hexdigest()[:12],
                    "Status": "RECEIVED",
                })
                ack_payload = json.dumps({"status": "OK", "carrier_idx": carrier_idx}).encode("utf-8")
                send_packet(conn, PKT_ACK, carrier_idx, ack_payload)

            elif pkt_type == PKT_END:
                packet_logs.append({
                    "Step": len(packet_logs) + 1,
                    "Direction": f"{client_addr_str} -> {server_addr_str}",
                    "Packet Type": "END (0x03)",
                    "Carrier": "FIN Signal",
                    "Size (KB)": round(frame_bytes / 1024.0, 3),
                    "Latency (ms)": round(pkt_ms, 2),
                    "SHA-256 (12H)": "-",
                    "Status": "SESSION CLOSED",
                })
                break

            else:
                raise RuntimeError(f"Unknown packet type: 0x{pkt_type:02x}")

        if metadata is None:
            raise RuntimeError("Session ended without receiving metadata.")

        num_carriers = int(metadata["num_carriers"])
        carrier_names = metadata.get("carrier_names", [f"Carrier {i + 1}" for i in range(num_carriers)])

        received_stegos: List[Optional[np.ndarray]] = []
        dropped_carriers: List[int] = []

        for k in range(num_carriers):
            if k in received_map:
                received_stegos.append(received_map[k])
            else:
                received_stegos.append(None)
                dropped_carriers.append(k)
                packet_logs.append({
                    "Step": len(packet_logs) + 1,
                    "Direction": f"{client_addr_str} -X-> {server_addr_str}",
                    "Packet Type": "CARRIER MISSING",
                    "Carrier": f"Carrier {k + 1} ({carrier_names[k]})",
                    "Size (KB)": 0.0,
                    "Latency (ms)": 0.0,
                    "SHA-256 (12H)": "DROPPED",
                    "Status": "DROPPED (ERASED)",
                })

        # Blind MC-SS extraction using the server's locally configured secret key
        extract_t0 = time.perf_counter()
        receiver_engine = MCSSStego(
            secret_key=self.secret_key,
            num_carriers=num_carriers,
            alpha=float(metadata["alpha"]),
            use_ecc=bool(metadata["use_ecc"]),
        )
        extracted_text, extracted_bits, stats = receiver_engine.extract_text(
            received_stegos, metadata, dropped_carriers=dropped_carriers
        )
        extract_ms = (time.perf_counter() - extract_t0) * 1000.0
        total_ms = (time.perf_counter() - session_start) * 1000.0

        session_result = {
            "success": True,
            "client_addr": client_addr_str,
            "server_addr": server_addr_str,
            "metadata": metadata,
            "received_stegos": received_stegos,
            "received_count": num_carriers - len(dropped_carriers),
            "dropped_carriers": dropped_carriers,
            "extracted_text": extracted_text,
            "extracted_bits": extracted_bits,
            "extraction_stats": stats,
            "packet_logs": packet_logs,
            "total_bytes_received": total_bytes_received,
            "extract_ms": round(extract_ms, 2),
            "total_ms": round(total_ms, 2),
            "server_key_used": self.secret_key,
        }

        self.last_session = session_result
        self.sessions.append(session_result)

        # Send final acknowledgment to client
        final_ack = json.dumps({
            "status": "DECODED",
            "received_count": session_result["received_count"],
            "dropped_carriers": dropped_carriers,
            "extract_ms": session_result["extract_ms"],
        }).encode("utf-8")
        send_packet(conn, PKT_ACK, NO_CARRIER, final_ack)

        if self.on_session_complete is not None:
            self.on_session_complete(session_result)


class StegoTCPClient:
    """
    TCP/IP Client that transmits MC-SS stego carrier images over a real TCP socket
    to a StegoTCPServer, with optional simulated network attacks (packet drop,
    AWGN noise, JPEG compression, or cropping in transit).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5000, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def send_carriers(
        self,
        stegos: List[np.ndarray],
        metadata: Dict[str, Any],
        carrier_names: Optional[List[str]] = None,
        drop_carriers: Optional[List[int]] = None,
        channel_attack: str = "None",
        attack_param: float = 0.0,
    ) -> Dict[str, Any]:
        """Open a TCP connection and transmit metadata + carrier image frames."""
        dropped_set = set(drop_carriers) if drop_carriers else set()
        num_carriers = len(stegos)
        names = carrier_names if carrier_names else [f"Carrier {i + 1}" for i in range(num_carriers)]

        wire_metadata = dict(metadata)
        wire_metadata["carrier_names"] = names
        if isinstance(wire_metadata.get("img_shape"), tuple):
            wire_metadata["img_shape"] = list(wire_metadata["img_shape"])

        tx_logs: List[Dict[str, Any]] = []
        total_bytes_sent = 0
        tx_start = time.perf_counter()

        with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
            local_addr = f"{sock.getsockname()[0]}:{sock.getsockname()[1]}"
            remote_addr = f"{sock.getpeername()[0]}:{sock.getpeername()[1]}"

            # 1. Send metadata header (never includes the secret key)
            t0 = time.perf_counter()
            meta_bytes = json.dumps(wire_metadata).encode("utf-8")
            sent = send_packet(sock, PKT_METADATA, NO_CARRIER, meta_bytes)
            total_bytes_sent += sent
            ack_type, _, _ = recv_packet(sock)
            rtt_ms = (time.perf_counter() - t0) * 1000.0
            if ack_type != PKT_ACK:
                raise RuntimeError("Did not receive ACK for session metadata.")

            tx_logs.append({
                "Step": len(tx_logs) + 1,
                "Packet": "METADATA (0x01)",
                "Carrier": "Session Header (No Key)",
                "Size (KB)": round(sent / 1024.0, 2),
                "RTT (ms)": round(rtt_ms, 2),
                "Status": "ACKED",
            })

            # 2. Stream each carrier image over TCP
            for k in range(num_carriers):
                if k in dropped_set:
                    tx_logs.append({
                        "Step": len(tx_logs) + 1,
                        "Packet": "CARRIER (0x02)",
                        "Carrier": f"Carrier {k + 1} ({names[k]})",
                        "Size (KB)": 0.0,
                        "RTT (ms)": 0.0,
                        "Status": "DROPPED BY FIREWALL",
                    })
                    continue

                img_to_send = stegos[k].copy()
                wire_format = "PNG"
                jpeg_qf = 75

                if channel_attack == "AWGN Noise" and attack_param > 0:
                    img_to_send = apply_gaussian_noise(img_to_send, sigma=float(attack_param))
                elif channel_attack == "Cropping" and attack_param > 0:
                    img_to_send = apply_cropping(img_to_send, crop_fraction=float(attack_param))
                elif channel_attack == "JPEG Compression":
                    wire_format = "JPEG"
                    jpeg_qf = int(attack_param) if attack_param > 0 else 70

                payload = encode_image_for_wire(img_to_send, wire_format=wire_format, jpeg_quality=jpeg_qf)

                t0 = time.perf_counter()
                sent = send_packet(sock, PKT_CARRIER, k, payload)
                total_bytes_sent += sent
                ack_type, _, _ = recv_packet(sock)
                rtt_ms = (time.perf_counter() - t0) * 1000.0
                if ack_type != PKT_ACK:
                    raise RuntimeError(f"Did not receive ACK for Carrier {k}.")

                tx_logs.append({
                    "Step": len(tx_logs) + 1,
                    "Packet": f"CARRIER {wire_format} (0x02)",
                    "Carrier": f"Carrier {k + 1} ({names[k]})",
                    "Size (KB)": round(sent / 1024.0, 2),
                    "RTT (ms)": round(rtt_ms, 2),
                    "Status": "DELIVERED & ACKED",
                })

            # 3. Send End-of-Transmission packet and wait for server decode ACK
            t0 = time.perf_counter()
            sent = send_packet(sock, PKT_END, NO_CARRIER, b"")
            total_bytes_sent += sent
            ack_type, _, final_ack_bytes = recv_packet(sock)
            rtt_ms = (time.perf_counter() - t0) * 1000.0
            final_ack = json.loads(final_ack_bytes.decode("utf-8")) if ack_type == PKT_ACK else {}

            tx_logs.append({
                "Step": len(tx_logs) + 1,
                "Packet": "END (0x03)",
                "Carrier": "FIN Signal",
                "Size (KB)": round(sent / 1024.0, 3),
                "RTT (ms)": round(rtt_ms, 2),
                "Status": f"SERVER {final_ack.get('status', 'DONE')}",
            })

        total_ms = (time.perf_counter() - tx_start) * 1000.0
        return {
            "local_addr": local_addr,
            "remote_addr": remote_addr,
            "total_bytes_sent": total_bytes_sent,
            "total_ms": round(total_ms, 2),
            "tx_logs": tx_logs,
            "server_ack": final_ack,
        }


def run_loopback_tcp_transfer(
    stegos: List[np.ndarray],
    metadata: Dict[str, Any],
    server_secret_key: str,
    carrier_names: List[str],
    host: str = "127.0.0.1",
    port: int = 0,
    drop_carriers: Optional[List[int]] = None,
    channel_attack: str = "None",
    attack_param: float = 0.0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Execute a complete live TCP/IP Client -> Server transfer on loopback.
    Spins up a real StegoTCPServer, connects a real StegoTCPClient, transfers framed
    carrier packets over TCP sockets, waits for server blind extraction, and cleanly
    closes the server socket.
    """
    done_event = threading.Event()
    server_result_holder: Dict[str, Any] = {}

    def _on_complete(res: Dict[str, Any]) -> None:
        server_result_holder.update(res)
        done_event.set()

    server = StegoTCPServer(
        host=host,
        port=port,
        secret_key=server_secret_key,
        on_session_complete=_on_complete,
    )
    bound_host, bound_port = server.start(background=True)

    try:
        client = StegoTCPClient(host=bound_host, port=bound_port)
        client_result = client.send_carriers(
            stegos=stegos,
            metadata=metadata,
            carrier_names=carrier_names,
            drop_carriers=drop_carriers,
            channel_attack=channel_attack,
            attack_param=attack_param,
        )
        done_event.wait(timeout=5.0)
        if not server_result_holder and server.last_session:
            server_result_holder.update(server.last_session)
    finally:
        server.stop()

    return client_result, server_result_holder
