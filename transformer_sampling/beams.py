import torch
import einops

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

    def __getitem__(self, batch_idx: int) -> "Beam":
        return Beam(self.model, self.tokenizer, self.logprob_sums[batch_idx], self.tokens[batch_idx, :])

    def get_logprobs_and_completions(self) -> list[tuple[float, str]]:
        return [(logprob_sum, self.tokenizer.decode(tokens))
        for logprob_sum, tokens in zip(self.logprob_sums, self.tokens)]

    def generate(
        self,
        num_beams: int
    ) -> "Beam":
        logprobs = self.model(self.tokens).logits[:, -1, :].log_softmax(-1)
        topk_logprobs, topk_toks = logprobs.topk(k=num_beams, dim=-1) 
        new_logprob_sums = einops.repeat(self.logprob_sums, "b -> b k", k=num_beams) + topk_logprobs # num_beams, num_beams (after 1st iteration)
        new_tokens = torch.cat([einops.repeat(self.tokens, "b s -> b k s", k=num_beams), topk_toks.unsqueeze(-1)], dim=-1) # num_beams num_beams seq_len (after 1st iteration)
        return Beam(self.model, self.tokenizer, new_logprob_sums.flatten(), new_tokens.flatten(0, 1))

    def filter(
        self, 
        num_beams: int
    ) -> "Beam":
        top_beam_indices = self.logprob_sums.topk(k=num_beams).indices.tolist()
        top_logprob_sums = self.logprob_sums[top_beam_indices]
        top_beams = self.tokens[top_beam_indices]

        return Beam(self.model, self.tokenizer, top_logprob_sums, top_beams)