import re
import json
import torch
import numpy as np
import pandas as pd

from transformers import AutoTokenizer
from dotenv import load_dotenv

from tqdm import tqdm
from torch import Tensor
from typing import Dict, Tuple, List, Optional, Union
from server import Server
from collections import Counter
from prompts import (
    SYSTEM_PROMPT,
    EVAL_SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLE_1_USER,
    FEW_SHOT_EXAMPLE_1_ASSISTANT,
    FEW_SHOT_EXAMPLE_2_USER,
    FEW_SHOT_EXAMPLE_2_ASSISTANT
)

load_dotenv()

class SEvRArgs:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    port: int = 30000
    num_samples: int = 11
    num_beams: int = 3
    lambda_k: int = 0.7
    tau: float = 0.6
    og_tau: float = 0.6
    alpha: float = 0.8
    sampling_temp: float = 0.9

class SEvR:
    """ Self-Evaluated Guided Beam Search for Reasoning (SEvR or pronounded sever)
    """
    def __init__(self, args: SEvRArgs) -> None:
        self.model_name = args.model_name
        self.tau = args.tau
        self.args = args

        self.__post_init__()

    def __post_init__(self):
        """ Initialize tokenkizer and sglang server"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.server = Server(self.model_name, self.args.port)

    def _format_prompt(
        self,
        prompt: str,
        tokenize: bool,
        add_gen_prompt: bool
    ) -> Union[List[int], List[str]]:
        """Format a standard prompt with system message and user input."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_gen_prompt
        )

    def _format_eval_prompt(
        self,
        prompt: str,
        tokenize: bool,
        add_gen_prompt: bool
    ) -> Union[List[int], List[str]]:
        """Format an evaluation prompt with few-shot examples."""
        few_shot_examples = [
            {"role": "user", "content": FEW_SHOT_EXAMPLE_1_USER},
            {"role": "assistant", "content": FEW_SHOT_EXAMPLE_1_ASSISTANT},
            {"role": "user", "content": FEW_SHOT_EXAMPLE_2_USER},
            {"role": "assistant", "content": FEW_SHOT_EXAMPLE_2_ASSISTANT},
        ]
        system_message = {"role": "system", "content": EVAL_SYSTEM_PROMPT}
        user_message = {
            "role": "user",
            "content": f"{prompt} Is the above step of reasoning: # (A) Correct # (B) Incorrect"
        }
    
        messages = [system_message] + few_shot_examples + [user_message]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_gen_prompt
        )

    def _to_tokens(self, prompt: str):
        return self.tokenizer.encode(prompt)

    def _from_tokens(self, tokens: List[int]):
        return self.tokenizer.decode(tokens, skip_special_tokens=False)

    def _decay_tau(self):
        self.tau = self.args.alpha * self.tau

    def _reset_tau(self):
        self.tau = self.args.og_tau

    def _is_final_answer_present(self, response: str) -> bool:
        """ Check if \boxed{}, which shows the presence of a final answer is present """
        pattern = r'\\boxed\{([^}]*)\}'
        match = re.search(pattern, response)
        if match:
            return True
        return False

    def _get_final_answer(self, response: str) -> int:
        """ Retrieve the final answer present or return -1 if it isn't """
        pattern = r'\\boxed\{([^}]*)\}'
        match = re.search(pattern, response)
        if match:
            boxed_content = match.group(1)
            return boxed_content
        else:
            return -1

    def generate(
        self,
        tokens: List[int],
        num_generations: int = None,
        stop_at_newline: bool = False,
        eval_prompt: bool = False,
        sampling_temp: float = 0.6
    ) -> Union[list, dict]:
        """ 
            Run inference for a list of tokens 
            Return type could either be a dict (for evaluation prompt) or list (for generation prompt)
        """
        try:
            return self.server.inference(
                prompt=tokens,
                num_generations=num_generations,
                stop_at_newline=stop_at_newline,
                eval_prompt=eval_prompt,
                sampling_temp=sampling_temp
            ) 
        except Exception as e:
            self.server.shutdown()
            raise

    def reason(self, prompt: str) -> None:
        num_iter = 0
        beams = [{"text": prompt, "meta_info": {"step_scores": [], "is_complete": False}}]
        try:
            while True:
                new_beams = []
                for beam in beams:
                    if beam["meta_info"]["is_complete"]:
                        new_beams.append(beam)
                        continue

                    tokens = self._format_prompt(beam["text"], True, True)
                    responses = self.generate(
                        tokens=tokens,
                        num_generations=self.args.num_samples,
                        stop_at_newline=True,
                        sampling_temp=self.args.sampling_temp
                    )
                    responses = self.append_input_str(responses, beam["text"])
                for res in responses:
                    is_complete = self._is_final_answer_present(res["text"])
                    confidence_score = self._get_confidence_score(res["text"])
                    lm_score = self._compute_lm_score(res["meta_info"]["output_token_logprobs"])

                    step_scores = beam["meta_info"]["step_scores"].copy()
                    step_scores.append((lm_score, confidence_score))
                
                    res["meta_info"]["step_scores"] = step_scores
                    res["meta_info"]["is_complete"] = is_complete
                    new_beams.append(res)

                self._decay_tau()
                beams = self._prune_beam(responses)
                num_iter += 1
                if all(beam["meta_info"]["is_complete"] for beam in beams) or num_iter == 10:
                    break

            self._reset_tau()
            final_res = [self._get_final_answer(beam["text"]) for beam in beams]
            return final_res, beams, num_iter
        except Exception as e:
            self.server.shutdown()
            raise

    def _prune_beam(self, responses: Dict):
        accumulated_log_scores = []
        lambda_k = self.args.lambda_k
        tau = self.tau
        num_beams = self.args.num_beams
        epsilon = 1e-8

        for res in responses:
            step_scores = res["meta_info"].get("step_scores", None)
            log_score = 0.0
            for lm_score, conf_score in step_scores:
                # Convert confidence scores to log-space, lm score is already sum of log probs
                log_lm = lm_score
                log_conf = np.log(max(conf_score, 1e-10))  
                log_score += (lambda_k * log_lm) + ((1 - lambda_k) * log_conf)
            accumulated_log_scores.append(log_score)

        # Temperature scaling and sampling
        logits = np.array(accumulated_log_scores) / tau
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()

        probs += epsilon
        probs /= probs.sum()

        assert len(responses) == probs.shape[0], "They should be equlal"
        
        sampled_indices = np.random.choice(
            len(responses),
            size=num_beams,
            replace=False,
            p=probs
        )
        sampled_responses = [responses[i] for i in sampled_indices]
        return sampled_responses

    def _compute_lm_score(
        self,
        output_token_logprobs: List[List[Union[float, int, None]]]
    ) -> float:
        """ Computes LM score by summing up the log probs across the output tokens """
        logprobs = [logprob for logprob, _, _ in output_token_logprobs]
        lm_score = sum(logprobs)
        return lm_score

    def append_input_str(self, responses: List[Dict], input_str: str) -> List[Dict]:
        for response in responses:
            response["text"] = input_str + response["text"]
        return responses

    def _get_confidence_score(self, prompt: str) -> float:
        """Generate eval response and find the relevant logprobs"""
        tokens = self._format_eval_prompt(prompt, True, True)
        response = self.generate(
            tokens=tokens,
            stop_at_newline=False,
            eval_prompt=True
        )
        output_tokens, token_ids_logprobs = self._parse_logprob_data(response)

        score = 0.0
        if "(A)" in response["text"]:
            score = np.exp(self._find_relevant_logprob(output_tokens, token_ids_logprobs, [4346, 32]))
        elif "(B)" in response["text"]:
            score = np.exp(self._find_relevant_logprob(output_tokens, token_ids_logprobs, [5349, 33], is_b=True))

        return score

    def _parse_logprob_data(self, response):
        """Extracts tokens and their corresponding logprob lists."""
        output_token_logprobs = response["meta_info"]["output_token_logprobs"]
        output_tokens = [tok for (_, tok, _) in output_token_logprobs]
        token_ids_logprobs = response["meta_info"]["output_token_ids_logprobs"]

        assert len(output_tokens) == len(token_ids_logprobs), "Mismatch between token and logprob list lengths"
        return output_tokens, token_ids_logprobs

    def _find_relevant_logprob(self, output_tokens, token_ids_logprobs, target_ids, is_b=False):
        """
        Finds the first matching token in `target_ids` and returns its corresponding logprob.
    
        If `is_b=True`, it maps:
            5349 -> 4346
            33   -> 32
        and returns the logprob of the mapped (A) token.
        """
        mapping = {
            5349: 4346,
            33: 32
        }
        for token_id in target_ids:
            if token_id in output_tokens:
                idx = output_tokens.index(token_id)
                # Determine which token ID we should retrieve the logprob for
                lookup_id = mapping[token_id] if is_b and token_id in mapping else token_id
                # Now find the logprob for the desired token ID at this position
                candidates = token_ids_logprobs[idx]
                for logprob, tid, _ in candidates:
                    if tid == lookup_id:
                        return logprob
        print("None of the target token IDs were found.")
        return -5

    @property
    def shutdown(self):
        self.server.shutdown


if __name__ == "__main__":
    args = SEvRArgs()
    self_eval_reasoner = SEvR(args)

    df = pd.read_csv("samples.csv")
    total_rows = len(df)

    output = []

    for idx, row in tqdm(df.iterrows(), desc="Processing", total=total_rows):
        question = row["question"]
        res, beams, num_iter = self_eval_reasoner.reason(question)
        if res:
            most_common = Counter(res).most_common(1)[0][0]
            is_correct = (most_common == row["answer"])
        else:
            is_correct = False

        output.append({
            "index": idx,
            "num_iter": num_iter,
            "is_correct": is_correct,
            "most_common": most_common if res else None,
            "ground_truth": row["answer"],
            "beams": beams,
        })
        with open(f"results/results_qwen2.5-7B_num_samples={args.num_samples}_num_beams={args.num_beams}.json", 'w') as f:
            json.dump(output, f, indent=2)

    self_eval_reasoner.shutdown
