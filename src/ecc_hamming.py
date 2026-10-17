"""
src/ecc_hamming.py
Message preprocessing, Hamming(7,4) Error-Correcting Code (ECC),
and carrier-level bit interleaving/de-interleaving.
"""

import numpy as np
from typing import Tuple, List, Optional


def text_to_bits(text: str) -> np.ndarray:
    """
    Convert a string into an array of bits (0 or 1), prepended with
    a 32-bit header encoding the message byte length.
    """
    raw_bytes = text.encode('utf-8')
    length = len(raw_bytes)
    
    # 32-bit big-endian length header
    header_bytes = length.to_bytes(4, byteorder='big')
    full_bytes = header_bytes + raw_bytes
    
    # Convert bytes to uint8 array and unpack bits
    byte_arr = np.frombuffer(full_bytes, dtype=np.uint8)
    bits = np.unpackbits(byte_arr).astype(np.int8)
    return bits


def bits_to_text(bits: np.ndarray) -> str:
    """
    Convert an array of bits back to a string, using the 32-bit length header.
    Handles corrupted characters gracefully using UTF-8 error replacement.
    """
    # Pack bits to bytes
    bits = np.asarray(bits, dtype=np.uint8)
    total_bytes = len(bits) // 8
    if total_bytes < 4:
        return ""
    
    # Pack up to the byte boundary
    usable_bits = bits[: total_bytes * 8]
    byte_arr = np.packbits(usable_bits)
    
    # Read length header
    payload_len = int.from_bytes(byte_arr[:4].tobytes(), byteorder='big')
    
    # Sanity check payload length
    if payload_len < 0 or payload_len > (total_bytes - 4):
        payload_bytes = byte_arr[4:].tobytes()
    else:
        payload_bytes = byte_arr[4 : 4 + payload_len].tobytes()
        
    try:
        return payload_bytes.decode('utf-8', errors='replace')
    except Exception:
        return payload_bytes.decode('latin-1', errors='replace')


class Hamming74:
    """
    Standard (7,4) Hamming Code encoder and syndrome/erasure decoder.
    Input: 4 data bits -> Output: 7 codeword bits.
    Can correct:
      - 1 bit inversion error per 7-bit block via syndrome decoding, OR
      - Up to 2 bit erasures (known missing bits) per 7-bit block.
    """

    # Parity check matrix H (3 x 7)
    # Columns represent binary values 1 to 7:
    # col 1 (001), col 2 (010), col 3 (011), col 4 (100),
    # col 5 (101), col 6 (110), col 7 (111)
    H = np.array([
        [0, 0, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1]
    ], dtype=np.int8)

    @classmethod
    def encode_block(cls, d: np.ndarray) -> np.ndarray:
        """
        Encode 4 data bits into 7 codeword bits.
        d: [d0, d1, d2, d3] mapped to indices [2, 4, 5, 6].
        Parity bits p1 (idx 0), p2 (idx 1), p3 (idx 3):
          p1 = d0 ^ d1 ^ d3
          p2 = d0 ^ d2 ^ d3
          p3 = d1 ^ d2 ^ d3
        """
        d0, d1, d2, d3 = d[0], d[1], d[2], d[3]
        p1 = d0 ^ d1 ^ d3
        p2 = d0 ^ d2 ^ d3
        p3 = d1 ^ d2 ^ d3
        return np.array([p1, p2, d0, p3, d1, d2, d3], dtype=np.int8)

    @classmethod
    def decode_block(cls, r: np.ndarray, erased_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, bool]:
        """
        Decode a 7-bit block.
        erased_indices: list of known corrupted/missing bit positions in [0..6].
        Returns (4_data_bits, corrected_flag).
        """
        r = r.copy().astype(np.int8)
        erased = list(erased_indices) if erased_indices is not None else []
        
        # If erasures are specified (up to 2 erasures can be corrected)
        if len(erased) == 1:
            # Single erasure: solve directly from any parity equation that involves it
            idx = erased[0]
            # Set to 0 and compute syndrome
            r[idx] = 0
            # Column in H corresponding to idx
            col = cls.H[:, idx]
            # Current syndrome
            s = np.dot(cls.H, r) % 2
            # If syndrome matches column, bit must be 1, else 0
            if np.array_equal(s, col):
                r[idx] = 1
            else:
                r[idx] = 0
            corrected = True
        elif len(erased) == 2:
            # Two erasures: solve 2x2 linear system over GF(2)
            idx1, idx2 = erased[0], erased[1]
            r[idx1] = 0
            r[idx2] = 0
            s = np.dot(cls.H, r) % 2
            col1 = cls.H[:, idx1]
            col2 = cls.H[:, idx2]
            # Try all 4 combinations (0,0), (0,1), (1,0), (1,1)
            found = False
            for b1 in (0, 1):
                for b2 in (0, 1):
                    if np.array_equal((b1 * col1 + b2 * col2) % 2, s):
                        r[idx1] = b1
                        r[idx2] = b2
                        found = True
                        break
                if found:
                    break
            corrected = found
        else:
            # Standard syndrome decoding (corrects 1 bit error)
            s = np.dot(cls.H, r) % 2
            syndrome_val = s[0] * 4 + s[1] * 2 + s[2] * 1
            corrected = False
            if syndrome_val > 0 and syndrome_val <= 7:
                # 1-indexed error position
                err_idx = syndrome_val - 1
                r[err_idx] ^= 1
                corrected = True

        data_bits = np.array([r[2], r[4], r[5], r[6]], dtype=np.int8)
        return data_bits, corrected

    @classmethod
    def encode(cls, bits: np.ndarray) -> np.ndarray:
        """
        Encode an arbitrary bitstream using Hamming(7,4).
        Pads bits with zeros to multiple of 4 if needed.
        Stores original unencoded bit count in a 16-bit prefix.
        """
        n_orig = len(bits)
        # Pad to multiple of 4
        pad_len = (4 - (n_orig % 4)) % 4
        if pad_len > 0:
            padded_bits = np.concatenate([bits, np.zeros(pad_len, dtype=np.int8)])
        else:
            padded_bits = bits.astype(np.int8)
            
        n_blocks = len(padded_bits) // 4
        blocks_4 = padded_bits.reshape((n_blocks, 4))
        
        encoded_blocks = []
        for i in range(n_blocks):
            encoded_blocks.append(cls.encode_block(blocks_4[i]))
            
        encoded_bits = np.concatenate(encoded_blocks)
        return encoded_bits

    @classmethod
    def decode(cls, encoded_bits: np.ndarray, orig_length: Optional[int] = None,
               erased_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Decode encoded bits back to original bitstream.
        erased_mask: boolean array of same length as encoded_bits where True indicates erased bit.
        """
        n_blocks = len(encoded_bits) // 7
        usable_len = n_blocks * 7
        usable_bits = encoded_bits[:usable_len].reshape((n_blocks, 7))
        
        if erased_mask is not None and len(erased_mask) >= usable_len:
            erased_blocks = erased_mask[:usable_len].reshape((n_blocks, 7))
        else:
            erased_blocks = np.zeros((n_blocks, 7), dtype=bool)

        decoded_blocks = []
        for i in range(n_blocks):
            erased_idx = np.where(erased_blocks[i])[0].tolist()
            d, _ = cls.decode_block(usable_bits[i], erased_indices=erased_idx)
            decoded_blocks.append(d)
            
        decoded_bits = np.concatenate(decoded_blocks)
        if orig_length is not None and orig_length <= len(decoded_bits):
            return decoded_bits[:orig_length]
        return decoded_bits


class BitInterleaver:
    """
    Distributes encoded bits across N carriers so that losing 1 carrier
    causes at most 1 (or 2) bit erasures per 7-bit codeword rather than wiping
    out a contiguous sequence of words.
    """

    @staticmethod
    def interleave(encoded_bits: np.ndarray, num_carriers: int) -> Tuple[List[np.ndarray], int]:
        """
        Interleave 7-bit codewords across num_carriers.
        Returns:
            sub_messages: List of N bit arrays of equal length.
            padded_total_len: Total length before splitting (for reconstruction).
        """
        n_bits = len(encoded_bits)
        # Pad to multiple of num_carriers
        pad = (num_carriers - (n_bits % num_carriers)) % num_carriers
        if pad > 0:
            padded = np.concatenate([encoded_bits, np.zeros(pad, dtype=np.int8)])
        else:
            padded = encoded_bits.astype(np.int8)

        # Block interleaving: write row-wise, read column-wise
        # Matrix shape: (L, num_carriers) where L = len(padded) // num_carriers
        L = len(padded) // num_carriers
        matrix = padded.reshape((L, num_carriers))
        
        # Carrier k gets column k
        sub_messages = [matrix[:, k].copy() for k in range(num_carriers)]
        return sub_messages, len(padded)

    @staticmethod
    def deinterleave(sub_messages: List[np.ndarray], orig_len: int,
                     dropped_carriers: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct the encoded bitstream from N sub-messages.
        dropped_carriers: list of carrier indices that were lost/zeroed.
        Returns:
            recovered_bits: 1D array of recovered bits.
            erased_mask: boolean mask indicating which bits came from dropped carriers.
        """
        num_carriers = len(sub_messages)
        L = len(sub_messages[0])
        matrix = np.zeros((L, num_carriers), dtype=np.int8)
        erased_matrix = np.zeros((L, num_carriers), dtype=bool)

        dropped_set = set(dropped_carriers) if dropped_carriers is not None else set()

        for k in range(num_carriers):
            matrix[:, k] = sub_messages[k][:L]
            if k in dropped_set:
                erased_matrix[:, k] = True

        flat_bits = matrix.flatten()[:orig_len]
        flat_erased = erased_matrix.flatten()[:orig_len]
        return flat_bits, flat_erased
