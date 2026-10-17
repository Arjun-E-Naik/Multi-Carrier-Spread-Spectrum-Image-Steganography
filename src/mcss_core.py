"""
src/mcss_core.py
Multi-Carrier Spread Spectrum (MC-SS) and Single-Image DSSS Steganography Engine.
Includes spatial embedding, high-pass correlation extraction, and message reconstruction.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional

from src.pn_generator import MultiCarrierPNGenerator
from src.ecc_hamming import Hamming74, BitInterleaver, text_to_bits, bits_to_text


class MCSSStego:
    """
    Multi-Carrier Spread Spectrum (MC-SS) Steganography Engine.
    Divides the secret payload across N cover images using orthogonal PN sequences.
    """

    def __init__(self, secret_key: str | int, num_carriers: int = 4, alpha: float = 2.0, use_ecc: bool = True):
        """
        Parameters:
            secret_key: Secret passphrase/seed for LFSR PN generation.
            num_carriers: Number of carrier images (N).
            alpha: Embedding amplitude factor (controls stego imperceptibility vs robustness).
            use_ecc: Whether to apply Hamming(7,4) ECC and carrier interleaving.
        """
        self.secret_key = secret_key
        self.num_carriers = num_carriers
        self.alpha = float(alpha)
        self.use_ecc = use_ecc
        self.pn_gen = MultiCarrierPNGenerator(secret_key, num_carriers)

    def embed_text(self, covers: List[np.ndarray], secret_text: str) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Embed secret text across N cover images.
        
        Parameters:
            covers: List of N 2D grayscale images (or color images converted to float32).
            secret_text: Plaintext message to hide.
            
        Returns:
            stegos: List of N stego images (uint8).
            metadata: Encoding parameters required for exact blind extraction.
        """
        if len(covers) != self.num_carriers:
            raise ValueError(f"Expected {self.num_carriers} covers, got {len(covers)}")

        # Convert text to binary bits (including 32-bit length header)
        raw_bits = text_to_bits(secret_text)
        n_raw = len(raw_bits)

        if self.use_ecc:
            # 1. Hamming(7,4) encoding
            encoded_bits = Hamming74.encode(raw_bits)
            # 2. Interleave across N carriers
            sub_messages, padded_total_len = BitInterleaver.interleave(encoded_bits, self.num_carriers)
        else:
            padded_total_len = (self.num_carriers - (n_raw % self.num_carriers)) % self.num_carriers + n_raw
            if padded_total_len > n_raw:
                padded = np.concatenate([raw_bits, np.zeros(padded_total_len - n_raw, dtype=np.int8)])
            else:
                padded = raw_bits.copy()
            sub_len = len(padded) // self.num_carriers
            sub_messages = [padded[k * sub_len : (k + 1) * sub_len] for k in range(self.num_carriers)]

        bits_per_carrier = len(sub_messages[0])
        h, w = covers[0].shape[:2]
        total_pixels = h * w
        
        # Chips per bit (spreading factor M)
        chips_per_bit = total_pixels // bits_per_carrier
        if chips_per_bit < 1:
            raise ValueError(f"Payload too large: {bits_per_carrier} bits require more than {total_pixels} pixels.")

        stegos = []
        residuals = []
        
        for k in range(self.num_carriers):
            cover = covers[k].astype(np.float32)
            # If color, work on 2D luminance / grayscale
            is_color = (cover.ndim == 3 and cover.shape[2] == 3)
            if is_color:
                channel = cover[:, :, 0]  # Or grayscale
            else:
                channel = cover

            # Bipolar representation of sub-message bits: 0 -> -1, 1 -> +1
            b_bits = sub_messages[k]
            bipolar_bits = (2 * b_bits - 1).astype(np.float32)

            # Generate carrier PN sequence of length: bits_per_carrier * chips_per_bit
            pn_len = bits_per_carrier * chips_per_bit
            pn_seq = self.pn_gen.get_carrier_pn_sequence(k, pn_len)

            # Repeat each bit chips_per_bit times and multiply by PN sequence
            expanded_bits = np.repeat(bipolar_bits, chips_per_bit)
            spread_signal = self.alpha * (expanded_bits * pn_seq)

            # Embed spread signal in flattened cover pixels
            flat_cover = channel.flatten()
            flat_stego = flat_cover.copy()
            flat_stego[:pn_len] += spread_signal

            # Clip to valid 8-bit dynamic range [0, 255]
            stego_channel = np.clip(np.round(flat_stego.reshape((h, w))), 0, 255).astype(np.uint8)
            
            if is_color:
                stego_img = cover.astype(np.uint8).copy()
                stego_img[:, :, 0] = stego_channel
            else:
                stego_img = stego_channel

            stegos.append(stego_img)
            residuals.append((stego_channel.astype(np.float32) - channel).astype(np.float32))

        metadata = {
            'num_carriers': self.num_carriers,
            'bits_per_carrier': bits_per_carrier,
            'chips_per_bit': chips_per_bit,
            'n_raw_bits': n_raw,
            'padded_total_len': padded_total_len,
            'use_ecc': self.use_ecc,
            'alpha': self.alpha,
            'img_shape': (h, w)
        }
        return stegos, metadata

    def extract_text(self, stegos: List[Optional[np.ndarray]], metadata: Dict[str, Any],
                     dropped_carriers: Optional[List[int]] = None) -> Tuple[str, np.ndarray, Dict[str, Any]]:
        """
        Extract hidden message from N stego images using blind correlation detection.
        
        Parameters:
            stegos: List of N received stego images. If a carrier was lost, pass None or black image.
            metadata: Metadata dict returned during embedding.
            dropped_carriers: Indices of carriers that are missing or discarded.
            
        Returns:
            extracted_text: Decoded string.
            extracted_bits: Raw or decoded bit array.
            stats: Extraction statistics (correlations, error flags).
        """
        bits_per_carrier = metadata['bits_per_carrier']
        chips_per_bit = metadata['chips_per_bit']
        pn_len = bits_per_carrier * chips_per_bit
        use_ecc = metadata['use_ecc']
        padded_total_len = metadata['padded_total_len']
        n_raw_bits = metadata['n_raw_bits']

        dropped = set(dropped_carriers) if dropped_carriers is not None else set()
        recovered_sub_messages = []
        correlations_per_carrier = []

        for k in range(self.num_carriers):
            stego = stegos[k]
            if stego is None or k in dropped:
                dropped.add(k)
                # Mark as zero bits for dropped carrier
                recovered_sub_messages.append(np.zeros(bits_per_carrier, dtype=np.int8))
                correlations_per_carrier.append(np.zeros(bits_per_carrier, dtype=np.float32))
                continue

            if stego.ndim == 3 and stego.shape[2] == 3:
                channel = stego[:, :, 0].astype(np.float32)
            else:
                channel = stego.astype(np.float32)

            # Blind host interference cancellation:
            # High-pass filtering via local 3x3 box blur subtraction
            blurred = cv2.blur(channel, (3, 3))
            residual = channel - blurred
            flat_residual = residual.flatten()[:pn_len]

            # Regenerate carrier PN sequence
            pn_seq = self.pn_gen.get_carrier_pn_sequence(k, pn_len)

            # Despreading: compute correlation for each bit block
            # Reshape into (bits_per_carrier, chips_per_bit)
            res_blocks = flat_residual.reshape((bits_per_carrier, chips_per_bit))
            pn_blocks = pn_seq.reshape((bits_per_carrier, chips_per_bit))

            # Cross-correlation per bit: C_i = sum(residual * pn)
            corr_values = np.sum(res_blocks * pn_blocks, axis=1)
            correlations_per_carrier.append(corr_values)

            # Decision threshold: C_i >= 0 -> 1, C_i < 0 -> 0
            rec_bits = (corr_values >= 0).astype(np.int8)
            recovered_sub_messages.append(rec_bits)

        # De-interleaving and Error Correction
        if use_ecc:
            flat_encoded, erased_mask = BitInterleaver.deinterleave(
                recovered_sub_messages, padded_total_len, dropped_carriers=list(dropped)
            )
            decoded_bits = Hamming74.decode(flat_encoded, orig_length=n_raw_bits, erased_mask=erased_mask)
        else:
            decoded_bits = np.concatenate(recovered_sub_messages)[:n_raw_bits]

        extracted_text = bits_to_text(decoded_bits)

        stats = {
            'dropped_carriers': list(dropped),
            'correlations': correlations_per_carrier,
            'decoded_bit_count': len(decoded_bits)
        }
        return extracted_text, decoded_bits, stats


class SingleImageDSSS:
    """
    Conventional Single-Image Direct Sequence Spread Spectrum (DSSS) Baseline.
    Embeds the entire unsplit payload into a single cover image.
    Used for quantitative benchmarking against Multi-Carrier Spread Spectrum.
    """

    def __init__(self, secret_key: str | int, alpha: float = 2.0):
        self.secret_key = secret_key
        self.alpha = float(alpha)
        self.pn_gen = MultiCarrierPNGenerator(secret_key, num_carriers=1)

    def embed_text(self, cover: np.ndarray, secret_text: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        raw_bits = text_to_bits(secret_text)
        n_raw = len(raw_bits)

        h, w = cover.shape[:2]
        total_pixels = h * w
        chips_per_bit = total_pixels // n_raw
        if chips_per_bit < 1:
            raise ValueError(f"Payload ({n_raw} bits) exceeds cover capacity ({total_pixels} pixels).")

        cover_float = cover.astype(np.float32)
        is_color = (cover_float.ndim == 3 and cover_float.shape[2] == 3)
        channel = cover_float[:, :, 0] if is_color else cover_float

        bipolar_bits = (2 * raw_bits - 1).astype(np.float32)
        pn_len = n_raw * chips_per_bit
        pn_seq = self.pn_gen.get_carrier_pn_sequence(0, pn_len)

        expanded_bits = np.repeat(bipolar_bits, chips_per_bit)
        spread_signal = self.alpha * (expanded_bits * pn_seq)

        flat_cover = channel.flatten()
        flat_stego = flat_cover.copy()
        flat_stego[:pn_len] += spread_signal

        stego_channel = np.clip(np.round(flat_stego.reshape((h, w))), 0, 255).astype(np.uint8)
        if is_color:
            stego_img = cover.astype(np.uint8).copy()
            stego_img[:, :, 0] = stego_channel
        else:
            stego_img = stego_channel

        metadata = {
            'n_raw_bits': n_raw,
            'chips_per_bit': chips_per_bit,
            'pn_len': pn_len,
            'alpha': self.alpha,
            'img_shape': (h, w)
        }
        return stego_img, metadata

    def extract_text(self, stego: Optional[np.ndarray], metadata: Dict[str, Any]) -> Tuple[str, np.ndarray]:
        if stego is None:
            # 100% loss if carrier is missing
            n_raw = metadata['n_raw_bits']
            return "", np.zeros(n_raw, dtype=np.int8)

        n_raw = metadata['n_raw_bits']
        chips_per_bit = metadata['chips_per_bit']
        pn_len = metadata['pn_len']

        if stego.ndim == 3 and stego.shape[2] == 3:
            channel = stego[:, :, 0].astype(np.float32)
        else:
            channel = stego.astype(np.float32)

        blurred = cv2.blur(channel, (3, 3))
        residual = channel - blurred
        flat_residual = residual.flatten()[:pn_len]

        pn_seq = self.pn_gen.get_carrier_pn_sequence(0, pn_len)

        res_blocks = flat_residual.reshape((n_raw, chips_per_bit))
        pn_blocks = pn_seq.reshape((n_raw, chips_per_bit))

        corr_values = np.sum(res_blocks * pn_blocks, axis=1)
        rec_bits = (corr_values >= 0).astype(np.int8)
        extracted_text = bits_to_text(rec_bits)
        return extracted_text, rec_bits
