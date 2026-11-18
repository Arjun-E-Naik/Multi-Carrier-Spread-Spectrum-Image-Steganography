"""
tests/test_network_tcp.py
Integration tests for the real TCP/IP Client-Server MC-SS steganography layer.
"""

import unittest
import numpy as np

from src.ecc_hamming import text_to_bits
from src.mcss_core import MCSSStego
from src.metrics import compute_ber
from src.network_tcp import run_loopback_tcp_transfer


def _make_synthetic_covers(num_carriers: int = 4) -> list:
    covers = []
    for i in range(num_carriers):
        x = np.linspace(-5, 5, 512)
        y = np.linspace(-5, 5, 512)
        xx, yy = np.meshgrid(x, y)
        cov = 128 + 40 * np.sin(xx * (i + 1) * 0.5) * np.cos(yy * 0.5)
        covers.append(np.clip(cov, 0, 255).astype(np.uint8))
    return covers


class TestMCSSTCPNetwork(unittest.TestCase):
    def setUp(self):
        self.covers = _make_synthetic_covers(4)
        self.carrier_names = ["Lena", "Baboon", "Peppers", "Cameraman"]
        self.secret_text = "Live TCP/IP Multi-Carrier Spread Spectrum Payload."
        self.secret_key = "NetTestKey#2026"
        self.raw_bits = text_to_bits(self.secret_text)

        self.mcss = MCSSStego(secret_key=self.secret_key, num_carriers=4, alpha=2.0, use_ecc=True)
        self.stegos, self.metadata = self.mcss.embed_text(self.covers, self.secret_text)

    def test_01_clean_tcp_transfer(self):
        """All 4 carriers delivered over real TCP sockets -> 0% BER."""
        client_res, server_res = run_loopback_tcp_transfer(
            stegos=self.stegos,
            metadata=self.metadata,
            server_secret_key=self.secret_key,
            carrier_names=self.carrier_names,
            port=0,
        )
        self.assertTrue(server_res["success"])
        self.assertEqual(server_res["received_count"], 4)
        self.assertEqual(server_res["dropped_carriers"], [])
        self.assertEqual(server_res["extracted_text"], self.secret_text)
        self.assertEqual(compute_ber(self.raw_bits, server_res["extracted_bits"]), 0.0)
        self.assertGreater(client_res["total_bytes_sent"], 0)

    def test_02_tcp_transfer_with_dropped_carrier(self):
        """Dropping Carrier 1 on the network still recovers 100% of message via Hamming ECC."""
        _, server_res = run_loopback_tcp_transfer(
            stegos=self.stegos,
            metadata=self.metadata,
            server_secret_key=self.secret_key,
            carrier_names=self.carrier_names,
            port=0,
            drop_carriers=[1],
        )
        self.assertTrue(server_res["success"])
        self.assertEqual(server_res["received_count"], 3)
        self.assertEqual(server_res["dropped_carriers"], [1])
        self.assertIsNone(server_res["received_stegos"][1])
        self.assertEqual(server_res["extracted_text"], self.secret_text)
        self.assertEqual(compute_ber(self.raw_bits, server_res["extracted_bits"]), 0.0)

    def test_03_tcp_wrong_server_key(self):
        """Server with wrong key receives valid TCP images but gets ~50% random BER."""
        _, server_res = run_loopback_tcp_transfer(
            stegos=self.stegos,
            metadata=self.metadata,
            server_secret_key="UnauthorizedKey#999",
            carrier_names=self.carrier_names,
            port=0,
        )
        self.assertTrue(server_res["success"])
        self.assertEqual(server_res["received_count"], 4)
        ber = compute_ber(self.raw_bits, server_res["extracted_bits"])
        self.assertGreater(ber, 0.30)
        self.assertLess(ber, 0.70)

    def test_04_tcp_in_transit_jpeg_compression(self):
        """In-transit JPEG compression (QF=85) over TCP wire decodes cleanly."""
        _, server_res = run_loopback_tcp_transfer(
            stegos=self.stegos,
            metadata=self.metadata,
            server_secret_key=self.secret_key,
            carrier_names=self.carrier_names,
            port=0,
            channel_attack="JPEG Compression",
            attack_param=85.0,
        )
        self.assertTrue(server_res["success"])
        self.assertEqual(server_res["received_count"], 4)
        ber = compute_ber(self.raw_bits, server_res["extracted_bits"])
        self.assertLess(ber, 0.05)


if __name__ == "__main__":
    unittest.main()
