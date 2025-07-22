import argparse
import os
import csv
import json
import requests
import pandas as pd
import numpy as np
import logging
import sympy
from tqdm import tqdm
from dataclasses import dataclass
from transformers import AutoTokenizer
from typing import List, Optional
from prompt import STANDARD_PROMPT, COT_PROMPT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/baseline_log.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BaselineArgs:
    data_dir: str = "data/game_of_24_easy.csv"
    model_name: str = "Qwen/Qwen2.5-32B"
    temperature: float = 0.7
    max_new_tokens: int = 32
    max_gen_attempts: int = 50

class BaselineSolver:
    def __init__(self, args: BaselineArgs, method: str, num_iter: int):
        self.args = args
        self.method = method.upper()
        self.num_iter = num_iter
        self.data_df = pd.read_csv(args.data_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.generate_url = "http://localhost:30000/generate"
        self.flush_url = "http://localhost:30000/flush_cache"
        self.results_file = f"results/results_{self.method.lower()}_simple.csv"
        self._initialize_results_csv()

    def _initialize_results_csv(self) -> None:
        if not os.path.exists(self.results_file):
            with open(self.results_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["puzzle", "solved"])

    def _save_results_to_csv(self, puzzle: str, solved: bool) -> None:
        with open(self.results_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([puzzle, solved])

    def _validate_solution(self, response_text: str) -> bool:
        answer = response_text
        try:
            simplified = answer.split("=")[0]
            res = int(sympy.simplify(simplified))
            return res == 24
        except Exception as e:
            logger.error(f"Error sending request: {e}")
            raise

    def _prepare_payload(self, prompt: str) -> dict:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {
            "text": text,
            "sampling_params": {
                "temperature": self.args.temperature,
                "max_new_tokens": self.args.max_new_tokens,
            },
        }

    def _send_request(self, url: str, payload: Optional[dict] = None) -> requests.Response:
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response
        except Exception as e:
            logger.error(f"Error in request to {url}: {e}")
            raise

    def _deliver_payload(self, payload: dict) -> requests.Response:
        try:
            return self._send_request(url=self.generate_url, payload=payload)
        except Exception as e:
            logger.error(f"Error sending request: {e}")
            raise

    def run_baseline(self):
        average_acc = 0
        prompt_template = STANDARD_PROMPT if self.method == "IO" else COT_PROMPT

        for idx, row in tqdm(self.data_df.iterrows(), total=len(self.data_df), desc="Processing puzzles"):
            pid, puzzle = row["Rank"], row["Puzzle"]
            logger.info(f"Processing puzzle {pid}: {puzzle}")
            self._send_request(url=self.flush_url, payload=None)
            
            initial_numbers = [int(n) for n in puzzle.split()]
            valid_solutions = 0

            res_list = []

            for _ in range(self.num_iter):
                prompt = prompt_template.format(input=puzzle)
                payload = self._prepare_payload(prompt)

                for attempt in range(self.args.max_gen_attempts):
                    try:
                        response = self._deliver_payload(payload)
                        response_text = response.json().get("text", "")
                        res_list.append(self._validate_solution(response_text))
                        break
                    except Exception as e:
                        if attempt < self.args.max_gen_attempts - 1:
                            logger.warning(f"Attempt {attempt+1} failed: {e}. Retrying...")
                        else:
                            logger.error(f"Failed after {self.args.max_gen_attempts} attempts: {e}")
                            res_list.append(0.0)

            acc = sum(res_list) / len(res_list)
            average_acc += acc
            self._save_results_to_csv(puzzle, acc)

        accuracy = (average_acc / len(self.data_df)) * 100
        logger.info(f"{average_acc} / {len(self.data_df)} solutions found ({accuracy:.2f}%)")

        with open(f"results/results_{self.method.lower()}_summary.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["method", "num_iter", "accuracy"])
            writer.writerow([self.method, self.num_iter, accuracy])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline experiments for 24 Game")
    parser.add_argument("--method", type=str, choices=["IO", "CoT"], required=True, help="Prompting method: IO or CoT")
    parser.add_argument("--num_iter", type=int, default=5, help="Number of iterations per puzzle")
    args = parser.parse_args()

    baseline_args = BaselineArgs()
    solver = BaselineSolver(baseline_args, args.method, args.num_iter)
    solver.run_baseline()