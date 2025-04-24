from dataclasses import dataclass
from transformers import AutoTokenizer

@dataclass
class Beam:
    tokenizer: AutoTokenizer
    logprob_sums: int
    tokens: int

    