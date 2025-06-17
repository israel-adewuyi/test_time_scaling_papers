import re
import json
from dataclasses import dataclass
from typing import Union, List, Tuple

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from server import Server
from prompts import COT_SYSTEM_PROMPT

@dataclass
class CoTArgs:
    """ Arguments for running CoT baseline """
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    port: int = 30000
    num_samples: int = 1
    sampling_temp: float = 0.9
    max_tokens: int = 4096

class CoTExperiment:
    """ Chain of Thought reasoning on sample gsm8k problems """
    def __init__(self, args: CoTArgs):
        self.args = args
        self.server = Server(args.model_name, args.port)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def get_final_answer(self, response: str) -> str:
        """Retrieve the final answer from a response or return -1 if not found."""
        pattern = r'\\boxed\{([^}]*)\}'
        match = re.search(pattern, response)
        return match.group(1) if match else "-1"

    def format_prompt(
        self,
        prompt: str,
        tokenize: bool = True,
        add_gen_prompt: bool = True
    ) -> Union[List[int], List[str]]:
        """Format a prompt with system message and user input."""
        messages = [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_gen_prompt
        )

    def generate(
        self,
        tokens: Union[List[int], List[str]],
        num_generations: int,
        stop_at_newline: bool = False,
        sampling_temp: float = 0.6,
        max_tokens: int = 512
    ) -> dict:
        """Run inference for a list of tokens, returning sglang server response."""
        try:
            return self.server.inference(
                prompt=tokens,
                num_generations=num_generations,
                stop_at_newline=stop_at_newline,
                sampling_temp=sampling_temp,
                max_tokens=max_tokens
            )
        except Exception:
            self.shutdown
            raise

    def run_cot(self, prompt: str, ground_truth: str) -> Tuple[bool, str, int]:
        """Run Chain of Thought for a single prompt and return result details."""
        tokens = self.format_prompt(prompt, tokenize=True, add_gen_prompt=True)
        try:
            response = self.generate(
                tokens=tokens,
                num_generations=self.args.num_samples,
                stop_at_newline=True,
                sampling_temp=self.args.sampling_temp,
                max_tokens=self.args.max_tokens
            )
            answer = self.get_final_answer(response['text'])
            is_correct = answer == str(ground_truth)
            return is_correct, response['text'], response['meta_info']['completion_tokens']
        except Exception:
            self.shutdown
            raise

    def process_dataset(self, csv_path: str) -> None:
        """Process dataset and log results to JSON."""
        df = pd.read_csv(csv_path)
        output = []

        for idx, row in tqdm(df.iterrows(), desc="Processing", total=len(df)):
            question = row["question"]
            ground_truth = row["answer"]
            is_correct, response_text, num_tokens = self.run_cot(question, ground_truth)

            output.append({
                "index": idx,
                "question": question,
                "is_correct": is_correct,
                "predicted_answer": self.get_final_answer(response_text),
                "ground_truth": ground_truth,
                "response_text": response_text,
                "num_tokens": num_tokens
            })

        with open(f"results/cot_qwen2.5-7B_num_samples={self.args.num_samples}.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

    @property
    def shutdown(self) -> None:
        """Shutdown the server after processing."""
        self.server.shutdown

if __name__ == "__main__":
    args = CoTArgs()
    cot_experiment = CoTExperiment(args)
    cot_experiment.process_dataset("samples.csv")
    cot_experiment.shutdown
