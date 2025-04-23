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

    def generate(
        self,
        num_beams: int
    ):
        logprobs = self.model(self.tokens).logits[:, -1, :].log_softmax(-1)
        topk_logprobs, topk_toks = torch.topk(logprobs, k=num_beams, dim=-1)

        print(f"{'=' * 20} Before beam generation {'=' * 20}")
        print(f"Log probs sum tensor is {self.logprob_sums}")
        print(f"Log probs sum tensor has shape {self.logprob_sums.shape}")
        print(f"Tokens tensor is {self.tokens}")
        print(f"Tokens tensor has shape {self.tokens.shape}")

        print(f"{'=' * 20} During beam generation {'=' * 20} \n\n\n")
        print(f"Top k logprobs: {topk_logprobs}")
        print(f"Top k tokens: {topk_toks}")

        print(f"{'=' * 20} After beam generation {'=' * 20} \n\n\n")
        
        new_logprob_sums = einops.repeat(self.logprob_sums, "b -> 1 (b k)", k=num_beams) + topk_logprobs
        print(f"New log probs sum tensor is {new_logprob_sums.flatten()}")
        print(f"New log probs sum tensor has shape {new_logprob_sums.flatten().shape}")
        new_tokens = torch.cat([einops.repeat(self.tokens, "b s -> b k s", k=num_beams), topk_toks.unsqueeze(-1)], dim=-1)
        print(f"Result of concatting tokens after reshaping: {new_tokens.flatten(0,1)}")
        print(f"Shape of reshaped tensor: {new_tokens.flatten(0,1).shape}")

        return Beam(self.model, self.tokenizer, new_logprob_sums.flatten(), new_tokens.flatten(0, 1))

        
        # print(res)
        # print(type(res))
        # return "Got here in beam search dataclass"

    def filter():
        pass