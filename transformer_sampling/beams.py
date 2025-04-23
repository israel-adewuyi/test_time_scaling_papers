from torch import Tensor
from jaxtyping import Float, Int
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Beam:
    """Class representing a beam for beam search"""
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    logprob_sums: Float[Tensor, "batch"]
    tokens: Float[Tensor, "batch seq_len"]

    def __get_item__(self, batch_idx: int) -> Beams:
        return Beam(self.model, self.tokenizer, self.logprob_sums[batch_idx], self.tokens[batch_idx, :])

    def generate():
        pass

    def filter():
        pass