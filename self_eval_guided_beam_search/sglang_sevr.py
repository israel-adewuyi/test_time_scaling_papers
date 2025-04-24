import torch
import einops
from torch import Tensor
from typing import Dict, Tuple, List, Optional, Union
from server import Server
from beams import Beam
from llm_response import Response
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

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


    def _format_prompt(self, prompt: str, tokenize: bool, add_gen_prompt: bool) -> Union[List[int], List[str]]:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant, presented with mathematical problems. Reason step by step before outputting your answer."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_gen_prompt
        )
        return text

    def _format_eval_prompt(self, prompt: str, tokenize: bool, add_gen_prompt: bool) -> Union[List[int], List[str]]:
        few_shot_examples = [
            {"role": "user", "content": """Example 1 Question: Allison brought some CDs online. Each CD cost $7. There was an additional charge of $4 per order for shipping costs. The total bill came to $60. How many CDs did Allison buy? Answer: Each CD cost 7 dollars. Is the above step of reasoning: # (A) Correct # (B) Incorrect
"""},
            {"role": "assistant", "content": "The above step of reasoning is (A)"},
            {"role": "user", "content": """Example 2 Question: Allison brought some CDs online. Each CD cost $7. There was an additional charge of $4 per order for shipping costs. The total bill came to $60. How many CDs did Allison buy? Answer: Each CD cost 7 dollars. And there was an additional charge of 4 dollars. So the total cost of each CD is 7 + 4 = 11 dollars. Is the above step of reasoning: # (A) Correct # (B) Incorrect
"""},
            {"role": "assistant", "content": "The above step of reasoning is (B)"}
        ]
        messages = [
            {"role": "system", "content": "You are given a problem as well as an intermediate reasoning step of an Assistant. Is the reasoning step (A) Correct (B) Incorrect?"}
        ] + few_shot_examples + [
            {"role": "user", "content": f"{prompt}  Is the above step of reasoning: # (A) Correct # (B) Incorrect"}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_gen_prompt
        )
        return text
    def _to_tokens(self, prompt: str):
        return self.tokenizer.encode(prompt)

    def _from_tokens(self, tokens: List[int]):
        return self.tokenizer.decode(tokens, skip_special_tokens=False)

    def generate(self, prompt: str) -> None:
        tokens = self._format_prompt(prompt, True, True)
        # print(f"In generate, prompt is {prompt} and token is {tokens}")
        beam = Beam(self.tokenizer, torch.tensor([0.0]), [tokens])
        eval_string = f"Question: {prompt}\nAnswer: "
        try:
            while True:
                completions_logprobs, completions_tokens, completions_eval_string = [], [], []
                for sub_logprob_sum, sub_beam in zip(beam.logprob_sums, beam.tokens):
                    for _ in range(self.args.num_samples):
                        # print(f"In generate, prompt is {sub_beam}")
                        response = self.server.inference(sub_beam)
                        # print(response)
                        response = Response(response, sub_logprob_sum, sub_beam)
                        completions_logprobs.append(response.logprobs_sum)
                        completions_tokens.append(response.get_tokens(False))
                        completions_eval_string.append(self._to_tokens(eval_string) + response.get_tokens(True))
                    break
                print(f"Logprobs current sum: {completions_logprobs}")
                # print(f"Completion tokens {completions_tokens}")
                for idx, (res, eval_str) in enumerate(zip(completions_tokens, completions_eval_string)):
                    print(f"Completion {idx}")
                    # print(self._from_tokens(res))
                    # print(f"{'=' * 50}")
                    print(self._from_tokens(eval_str))
                    print(f"{'=' * 100}")
                self.evaluate(completions_eval_string)
                break
                
        except Exception as e:
            self.server.shutdown
            raise

    def evaluate(self, tokens_to_evaluate: List):
        samples = []
        for sample in tokens_to_evaluate:
            samples.append(self._format_eval_prompt(sample, True, True))

        for sample in samples:
            print(f"{'=' * 50}")
            response = self.server.inference(sample)
            print(response)
            # print(response["output_ids"])
            print(self._from_tokens(response["output_ids"]))
            print(f"{'=' * 50}")
        

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