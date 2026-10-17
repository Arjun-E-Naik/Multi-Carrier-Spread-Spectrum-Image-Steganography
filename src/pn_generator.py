"""
src/pn_generator.py
Linear Feedback Shift Register (LFSR) and Gold Code Pseudo-Noise (PN) Generator
for Multi-Carrier Spread Spectrum (MC-SS) Image Steganography.
"""

import hashlib
import numpy as np
from typing import Union, List, Tuple


class LFSR:
    """
    Galois/Fibonacci Linear Feedback Shift Register (LFSR) of configurable degree.
    Produces maximal-length sequences (m-sequences) with high autocorrelation peak
    and low off-peak correlation.
    """
    def __init__(self, taps: Tuple[int, ...], seed: int, length_bits: int = 16):
        """
        Parameters:
            taps: Feedback tap positions (1-indexed, e.g. (16, 14, 13, 11) for degree 16).
            seed: Initial non-zero register state.
            length_bits: Register bit-width (degree m).
        """
        self.taps = [t - 1 for t in taps]  # Convert to 0-indexed
        self.degree = length_bits
        self.mask = (1 << self.degree) - 1
        
        # Ensure non-zero seed
        initial_state = seed & self.mask
        if initial_state == 0:
            initial_state = 0xACE1 & self.mask
            if initial_state == 0:
                initial_state = 1
        self.state = initial_state

    def step(self) -> int:
        """Execute one LFSR shift and return the output bit (0 or 1)."""
        # Feedback is XOR sum of tap positions
        feedback = 0
        for tap in self.taps:
            feedback ^= (self.state >> tap) & 1
        
        output = self.state & 1
        self.state = ((self.state >> 1) | (feedback << (self.degree - 1))) & self.mask
        return output

    def generate(self, n_chips: int) -> np.ndarray:
        """Generate n_chips binary bits (0 or 1)."""
        seq = np.empty(n_chips, dtype=np.int8)
        for i in range(n_chips):
            seq[i] = self.step()
        return seq


class MultiCarrierPNGenerator:
    """
    Generates N distinct, near-orthogonal pseudo-noise sequences for MC-SS.
    Uses Gold Code construction or carrier-specific phase-shifted preferred m-sequences
    initialized from a shared secret key.
    """

    # Preferred pair of primitive polynomials for degree 16:
    # Poly 1: x^16 + x^14 + x^13 + x^11 + 1 (taps: 16, 14, 13, 11)
    # Poly 2: x^16 + x^15 + x^13 + x^4 + 1  (taps: 16, 15, 13, 4)
    TAPS_A = (16, 14, 13, 11)
    TAPS_B = (16, 15, 13, 4)
    DEGREE = 16

    def __init__(self, secret_key: Union[int, str], num_carriers: int = 4):
        """
        Parameters:
            secret_key: Secret passphrase or integer seed.
            num_carriers: Number of orthogonal carriers (N).
        """
        self.secret_key = secret_key
        self.num_carriers = num_carriers
        self.base_seed = self._derive_seed(secret_key)

    @staticmethod
    def _derive_seed(secret_key: Union[int, str]) -> int:
        """Derive a reproducible 32-bit positive integer seed from string or int."""
        if isinstance(secret_key, int):
            return abs(secret_key) & 0xFFFFFFFF
        # Hash string using SHA-256 for uniform entropy
        digest = hashlib.sha256(str(secret_key).encode('utf-8')).digest()
        seed = int.from_bytes(digest[:4], byteorder='big')
        return seed & 0xFFFFFFFF

    def get_carrier_pn_sequence(self, carrier_idx: int, length: int) -> np.ndarray:
        """
        Generate a bipolar (-1, +1) PN sequence for carrier_idx of length `length`.
        
        Gold code logic:
        Carrier k uses LFSR_A with base seed, and LFSR_B initialized with a shifted seed
        derived from (base_seed + carrier_idx * salt).
        The bitwise XOR produces the k-th Gold sequence:
            b_k(t) = a(t) ^ b(t + phase_k)
        Converted to bipolar format:
            p_k(t) = 2 * b_k(t) - 1  in {-1, +1}
        """
        if carrier_idx < 0 or carrier_idx >= self.num_carriers:
            raise ValueError(f"carrier_idx {carrier_idx} out of range [0, {self.num_carriers - 1}]")

        seed_a = (self.base_seed ^ 0x55555555) & 0xFFFF
        if seed_a == 0:
            seed_a = 0x1234

        # Generate unique non-zero seed and phase shift per carrier
        seed_b = ((self.base_seed >> 8) ^ ((carrier_idx + 1) * 0x9E3779B9)) & 0xFFFF
        if seed_b == 0:
            seed_b = 0x5678 + carrier_idx * 17

        lfsr_a = LFSR(self.TAPS_A, seed_a, self.DEGREE)
        lfsr_b = LFSR(self.TAPS_B, seed_b, self.DEGREE)

        # Advance LFSR_B by carrier-specific offset to ensure low cross-correlation
        offset = (carrier_idx * 1337) % 65535
        for _ in range(offset):
            lfsr_b.step()

        # Generate bits
        bits_a = lfsr_a.generate(length)
        bits_b = lfsr_b.generate(length)

        # Gold code: XOR combination
        gold_bits = bits_a ^ bits_b

        # Convert to bipolar: 0 -> -1, 1 -> +1
        bipolar_seq = (2 * gold_bits - 1).astype(np.float32)
        return bipolar_seq

    def get_all_carrier_sequences(self, length: int) -> List[np.ndarray]:
        """Generate bipolar PN sequences for all N carriers."""
        return [self.get_carrier_pn_sequence(k, length) for k in range(self.num_carriers)]

    def compute_cross_correlation(self, length: int = 10000) -> np.ndarray:
        """
        Compute normalized cross-correlation matrix between all carriers
        to verify near-orthogonality (off-diagonal values should be close to 0,
        diagonal values equal to 1.0).
        """
        seqs = self.get_all_carrier_sequences(length)
        corr_matrix = np.zeros((self.num_carriers, self.num_carriers), dtype=np.float32)
        for i in range(self.num_carriers):
            for j in range(self.num_carriers):
                corr = np.dot(seqs[i], seqs[j]) / length
                corr_matrix[i, j] = corr
        return corr_matrix
