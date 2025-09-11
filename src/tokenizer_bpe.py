# src/tokenizer_bpe.py
import json
from pathlib import Path
from typing import List, Tuple, Dict

from huggingface_hub import hf_hub_download


class BPETokenizer:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        # Load vocab.json
        vocab_path = hf_hub_download(repo_id=model_name, filename="vocab.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.encoder: Dict[str, int] = json.load(f)

        # Build reverse mapping
        self.decoder: Dict[int, str] = {v: k for k, v in self.encoder.items()}

        # Load merges.txt
        merges_path = hf_hub_download(repo_id=model_name, filename="merges.txt")
        merges: List[str] = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    merges.append(tuple(line.strip().split()))

        # Build ranking: pair -> priority
        self.bpe_ranks: Dict[Tuple[str, str], int] = {tuple(merge): i for i, merge in enumerate(merges)}

    def get_pairs(self, word: Tuple[str]) -> set[Tuple[str, str]]:
        """Return set of symbol pairs in a word."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def bpe(self, token: str) -> str:
        """Apply BPE to a single token (word)."""
        word = tuple(token)
        pairs = self.get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if i < len(word)-1 and word[i] == first and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)

        return " ".join(word)

    def encode(self, text: str) -> List[int]:
        """Tokenize text into token IDs."""
        tokens: List[int] = []
        for word in text.strip().split():
            bpe_tokens = self.bpe(word).split(" ")
            tokens.extend([self.encoder[tok] for tok in bpe_tokens if tok in self.encoder])
        return tokens

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back into text."""
        return "".join([self.decoder[i] for i in ids if i in self.decoder])
