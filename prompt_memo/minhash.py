"""MinHash shingling and Jaccard similarity estimation for fuzzy prompt matching."""

import hashlib
import json
import random
import re
import unicodedata
from typing import Sequence, Union


def compute_shingles(text: str, k: int = 3, mode: str = "char") -> set[str]:
    """Extracts a set of k-shingles from text.

    Args:
        text: Input string to shingle.
        k: Length of each shingle (default: 3).
        mode: 'char' for character n-grams, 'word' for word n-grams.

    Returns:
        A set of unique shingle strings. Empty text returns an empty set.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")
    if not text:
        return set()

    normalized = unicodedata.normalize("NFKC", text).lower().strip()
    if not normalized:
        return set()

    if mode == "word":
        words = re.findall(r"\w+", normalized)
        if not words:
            return set()
        if len(words) < k:
            return {" ".join(words)}
        return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}
    elif mode == "char":
        collapsed = re.sub(r"\s+", " ", normalized)
        if len(collapsed) < k:
            return {collapsed}
        return {collapsed[i : i + k] for i in range(len(collapsed) - k + 1)}
    else:
        raise ValueError(f"Unsupported shingle mode: '{mode}'. Must be 'char' or 'word'.")


def exact_jaccard(shingles_a: set[str], shingles_b: set[str]) -> float:
    """Calculates exact Jaccard similarity between two sets of shingles.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|.
    Returns 1.0 if both sets are empty, 0.0 if only one is empty.
    """
    if not shingles_a and not shingles_b:
        return 1.0
    if not shingles_a or not shingles_b:
        return 0.0
    intersection = len(shingles_a & shingles_b)
    union = len(shingles_a | shingles_b)
    return intersection / union if union > 0 else 0.0

def estimate_similarity(sig_a: Sequence[int], sig_b: Sequence[int]) -> float:
    """Estimates Jaccard similarity between two equal-length MinHash signatures.

    Returns the proportion of matching permutation hash values.
    """
    if not sig_a or not sig_b:
        return 0.0
    if len(sig_a) != len(sig_b):
        raise ValueError(
            f"Signatures must have identical length (got {len(sig_a)} and {len(sig_b)})."
        )
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def serialize_signature(signature: Sequence[int]) -> str:
    """Serializes a MinHash signature into a compact comma-separated string."""
    return ",".join(str(val) for val in signature)


def deserialize_signature(serialized: str) -> list[int]:
    """Deserializes a MinHash signature from comma-separated or JSON format."""
    trimmed = serialized.strip()
    if not trimmed:
        return []
    if trimmed.startswith("["):
        return [int(x) for x in json.loads(trimmed)]
    return [int(x) for x in trimmed.split(",") if x]


class MinHasher:
    """Deterministic Carter-Wegman MinHash generator for fuzzy text matching."""

    # 9th Mersenne prime: 2^61 - 1
    MERSENNE_PRIME_61 = (1 << 61) - 1

    def __init__(
        self,
        num_perm: int = 64,
        k: int = 3,
        mode: str = "char",
        seed: int = 42,
    ) -> None:
        """Initializes the MinHasher with deterministic permutation coefficients.

        Args:
            num_perm: Number of hash permutations (signature dimensions).
            k: Length of shingles.
            mode: Shingling mode ('char' or 'word').
            seed: Seed for reproducible permutation coefficients.
        """
        if num_perm <= 0:
            raise ValueError("num_perm must be a positive integer.")
        self.num_perm = num_perm
        self.k = k
        self.mode = mode
        self.seed = seed
        self._permutations = self._generate_permutations(num_perm, seed)

    @classmethod
    def _generate_permutations(cls, num_perm: int, seed: int) -> list[tuple[int, int]]:
        rng = random.Random(seed)
        p = cls.MERSENNE_PRIME_61
        permutations: list[tuple[int, int]] = []
        for _ in range(num_perm):
            a = rng.randint(1, p - 1)
            b = rng.randint(0, p - 1)
            permutations.append((a, b))
        return permutations

    def compute_shingles(self, text: str) -> set[str]:
        """Extracts shingles from text using the configured k and mode."""
        return compute_shingles(text, k=self.k, mode=self.mode)

    def compute_signature(self, text_or_shingles: Union[str, set[str]]) -> list[int]:
        """Generates a deterministic MinHash signature for the input text or shingles."""
        if isinstance(text_or_shingles, str):
            shingles = self.compute_shingles(text_or_shingles)
        else:
            shingles = text_or_shingles

        if not shingles:
            return [0] * self.num_perm

        p = self.MERSENNE_PRIME_61
        base_hashes = [
            int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "little")
            for s in shingles
        ]

        signature: list[int] = []
        for a, b in self._permutations:
            min_val = min((a * h + b) % p for h in base_hashes)
            signature.append(min_val)

        return signature

    def estimate_similarity(self, sig_a: Sequence[int], sig_b: Sequence[int]) -> float:
        """Estimates Jaccard similarity between two signatures."""
        return estimate_similarity(sig_a, sig_b)

    def similarity_of_texts(self, text_a: str, text_b: str) -> float:
        """End-to-end helper: computes signatures and estimates similarity of two texts."""
        sig_a = self.compute_signature(text_a)
        sig_b = self.compute_signature(text_b)
        return self.estimate_similarity(sig_a, sig_b)
