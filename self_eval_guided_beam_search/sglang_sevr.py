import torch
import einops
from torch import Tensor
from typing import Dict, Tuple, List
from server import Server
from beams import Beam
from llm_response import Response
from transformers import AutoModelForCausalLM, AutoTokenizer

class SEvRArgs:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "cuda:3"
    port: int = 30000
    num_samples: int = 5
    num_beams: int = 3

"""
1. Old tokens (current prompt) should be merged with new.
"""
class SEvR:
    def __init__(self, args: SEvRArgs) -> None:
        self.model_name = args.model_name
        self.device = args.device
        self.args = args

        self.__post_init__()

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.server = Server(self.model_name, self.args.port)


    def _to_tokens(self, prompt: str):
        return self.tokenizer.encode(prompt)

    def _from_tokens(self, tokens: List[int]):
        return self.tokenizer.decode(tokens, skip_special_tokens=False)

    def generate(self, prompt: str) -> None:
        tokens = self._to_tokens(prompt)
        beam = Beam(self.tokenizer, torch.tensor([0.0]), [tokens])

        try:
            while True:
                completions_logprobs, completions_tokens = [], []
                for sub_logprob_sum, sub_beam in zip(beam.logprob_sums, beam.tokens):
                    for _ in range(self.args.num_samples):
                        response = self.server.inference(sub_beam)
                        response = Response(response, sub_logprob_sum, sub_beam)
                        completions_logprobs.append(response.logprobs_sum)
                        completions_tokens.append(response.get_tokens)
                    break
                print(f"Logprobs current sum: {completions_logprobs}")
                # print(f"Completion tokens {completions_tokens}")
                for idx, res in enumerate(completions_tokens):
                    print(f"Completion {idx}")
                    print(self._from_tokens(res))
                    print()
                break
        except Exception as e:
            self.server.shutdown
            raise

    def evaluate(self, ):
        pass

    @property
    def shutdown(self):
        self.server.shutdown


if __name__ == "__main__":
    args = SEvRArgs()
    self_eval_reasoner = SEvR(args)
    
    # example_prompt = "What am I?"
    example_prompt = """
    Marilyn's 1st record sold 10 times as many copies as Harald's. If they sold 88,000 copies combined, how many copies did Harald sell?
"""
    toks = self_eval_reasoner.generate(example_prompt)
    self_eval_reasoner.shutdown