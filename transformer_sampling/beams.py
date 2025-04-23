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

    @property
    def get_logprobs_and_completions(self) -> list[tuple[float, str]]:
        return [(logprob_sum, self.tokenizer.decode(tokens))
        for logprob_sum, tokens in zip(self.logprob_sums, self.tokens)]

    def generate(
        self,
        num_beams: int
    ) -> "Beam":
        """
        Generate new beams by expanding the current beams with top-k predictions.
        """
        logprobs = self.model(self.tokens).logits[:, -1, :].log_softmax(-1)
        topk_logprobs, topk_toks = logprobs.topk(k=num_beams, dim=-1) 
        new_logprob_sums = einops.repeat(self.logprob_sums, "b -> b k", k=num_beams) + topk_logprobs # num_beams, num_beams (after 1st iteration)
        new_tokens = torch.cat([einops.repeat(self.tokens, "b s -> b k s", k=num_beams), topk_toks.unsqueeze(-1)], dim=-1) # num_beams num_beams seq_len (after 1st iteration)
        return Beam(self.model, self.tokenizer, new_logprob_sums.flatten(), new_tokens.flatten(0, 1))

    def filter(
        self, 
        num_beams: int
    ) -> tuple["Beam", "Beam"]:
        """
        Filter beams into continuing and terminated groups based on EOS token.
        """
        top_beam_indices = self.logprob_sums.topk(k=num_beams).indices.tolist()
        
        new_tokens = self.tokens[:, -1]
        terminated_indices = torch.nonzero(new_tokens == self.tokenizer.eos_token_id)

        best_continuing = [i for i in top_beam_indices if i not in terminated_indices]
        best_terminated = [i for i in top_beam_indices if i in terminated_indices]

        return self[best_continuing], self[best_terminated]