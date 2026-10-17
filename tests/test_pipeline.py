"""
tests/test_pipeline.py
Unit tests and pipeline verification for MC-SS Image Steganography.
"""

import os
import unittest
import numpy as np

from src.pn_generator import LFSR, MultiCarrierPNGenerator
from src.ecc_hamming import text_to_bits, bits_to_text, Hamming74, BitInterleaver
from src.mcss_core import MCSSStego, SingleImageDSSS
from src.attacks import apply_gaussian_noise, apply_jpeg_compression, apply_cropping, simulate_carrier_drop
from src.metrics import compute_psnr, compute_ssim, compute_ber


class TestMCSSPipeline(unittest.TestCase):

    def test_01_lfsr_and_gold_orthogonality(self):
        """Verify LFSR bipolar generation and low cross-correlation."""
        pn_gen = MultiCarrierPNGenerator(secret_key="TestSeed123", num_carriers=4)
        seqs = pn_gen.get_all_carrier_sequences(length=5000)
        self.assertEqual(len(seqs), 4)
        for s in seqs:
            # Check bipolar values
            self.assertTrue(np.all(np.isin(s, [-1.0, 1.0])))
            # Check near zero-mean
            self.assertAlmostEqual(np.mean(s), 0.0, delta=0.08)

        # Cross-correlation check
        corr = pn_gen.compute_cross_correlation(length=10000)
        for i in range(4):
            self.assertAlmostEqual(corr[i, i], 1.0, places=4)
            for j in range(4):
                if i != j:
                    self.assertLess(abs(corr[i, j]), 0.15, f"Cross-correlation {corr[i, j]} between {i} and {j} too high")

    def test_02_text_bits_conversion(self):
        """Verify string to bit array and back."""
        orig_text = "Steganography Test 2026! @#$%"
        bits = text_to_bits(orig_text)
        recovered = bits_to_text(bits)
        self.assertEqual(orig_text, recovered)

    def test_03_hamming_ecc_single_error_correction(self):
        """Verify Hamming(7,4) corrects 1-bit errors."""
        bits = np.array([1, 0, 1, 1], dtype=np.int8)
        encoded = Hamming74.encode_block(bits)
        self.assertEqual(len(encoded), 7)

        # Introduce 1-bit error at position 4
        corrupted = encoded.copy()
        corrupted[4] ^= 1

        decoded, corrected = Hamming74.decode_block(corrupted)
        self.assertTrue(corrected)
        np.testing.assert_array_equal(bits, decoded)

    def test_04_hamming_ecc_erasure_correction(self):
        """Verify Hamming(7,4) corrects 2 known erasures."""
        bits = np.array([0, 1, 1, 0], dtype=np.int8)
        encoded = Hamming74.encode_block(bits)

        # Erase positions 1 and 5
        corrupted = encoded.copy()
        corrupted[1] = 0
        corrupted[5] = 0

        decoded, corrected = Hamming74.decode_block(corrupted, erased_indices=[1, 5])
        self.assertTrue(corrected)
        np.testing.assert_array_equal(bits, decoded)

    def test_05_clean_embedding_and_psnr(self):
        """Verify clean embedding achieves PSNR > 40 dB and 0% BER."""
        # Synthetic 512x512 naturalistic covers
        covers = []
        for i in range(4):
            x = np.linspace(-5, 5, 512)
            y = np.linspace(-5, 5, 512)
            xx, yy = np.meshgrid(x, y)
            cov = 128 + 40 * np.sin(xx * (i + 1) * 0.5) * np.cos(yy * 0.5)
            covers.append(np.clip(cov, 0, 255).astype(np.uint8))

        secret = "Secret Multi-Carrier Payload"
        mcss = MCSSStego(secret_key="TestKey42", num_carriers=4, alpha=1.5, use_ecc=True)
        stegos, meta = mcss.embed_text(covers, secret)

        # Verify PSNR > 40 dB
        for k in range(4):
            p = compute_psnr(covers[k], stegos[k])
            s = compute_ssim(covers[k], stegos[k])
            self.assertGreater(p, 40.0, f"Carrier {k} PSNR {p} < 40 dB")
            self.assertGreater(s, 0.95, f"Carrier {k} SSIM {s} < 0.95")

        # Extract and verify 0% BER
        raw_bits = text_to_bits(secret)
        rec_txt, rec_bits, _ = mcss.extract_text(stegos, meta)
        ber = compute_ber(raw_bits, rec_bits)
        self.assertEqual(ber, 0.0, f"Clean BER is {ber}, expected 0.0")
        self.assertEqual(secret, rec_txt)

    def test_06_carrier_drop_resilience(self):
        """Verify dropping 1 carrier retains 100% text recovery with ECC."""
        covers = [np.full((512, 512), 120 + 10 * i, dtype=np.uint8) for i in range(4)]
        secret = "Resilient Carrier Drop Verification."
        mcss = MCSSStego(secret_key="KeyResilience", num_carriers=4, alpha=2.5, use_ecc=True)
        stegos, meta = mcss.embed_text(covers, secret)

        # Drop carrier 1
        stegos_dropped = simulate_carrier_drop(stegos, [1])
        rec_txt, rec_bits, _ = mcss.extract_text(stegos_dropped, meta, dropped_carriers=[1])
        raw_bits = text_to_bits(secret)
        ber = compute_ber(raw_bits, rec_bits)

        self.assertEqual(ber, 0.0, f"Expected 0% BER with 1 dropped carrier, got {ber}")
        self.assertEqual(secret, rec_txt)


if __name__ == '__main__':
    unittest.main()
