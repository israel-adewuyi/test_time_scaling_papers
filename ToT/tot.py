import json
import requests
import pandas as pd
import numpy as np

from tqdm import tqdm
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict
from pydantic import BaseModel, RootModel
from transformers import AutoTokenizer
from typing import Dict, List, Optional


from prompt import PROPOSE_PROMPT_24, VALUE_PROMPT_24

@dataclass
class ToTArgs:
    generations_per_step: int = 4
    total_steps: int = 3
    data_dir: str = "data/game_of_24.csv"
    model_name: str = "Qwen/Qwen2.5-32B"

    # BFS-specific arg
    breadth_limit: int = 2

class TreeOfThought(ABC):
    def __init__(self, args: ToTArgs):
        pass
    def generator(self): 
        pass
    def evaluator(self):
        pass


class TreeOfThoughtBFS(TreeOfThought):
    def __init__(self, args: ToTArgs):
        self.generations_per_step = args.generations_per_step
        self.total_steps = args.total_steps
        self.breadth_limit = args.breadth_limit
        self.model_name = args.model_name
        self.data_df = pd.read_csv(args.data_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.generate_url = "http://localhost:30000/generate"
        self.flush_url = "http://localhost:30000/flush_cache"
        
    def generator(self, state: str):
        payload = self._prepare_payload(state)

        response = self._deliver_payload(payload=payload)
        print(response.json()["text"])
        next_state_data = json.loads(response.json()["text"])
        operations, numbers_left = [], []

        for key, value in next_state_data.items():
            op, num_left = value["operation"], value["numbers_left"]
            num_left = " ".join(map(str, num_left))
            operations.append(op)
            numbers_left.append(num_left)
            
        return operations, numbers_left
   
    def evaluator(self, temp_next_states: List):
        next_state_values = []
        
        for next_state in tqdm(temp_next_states):
            value_prompt = VALUE_PROMPT_24.format(input_numbers=next_state)
            value_payload = self._prepare_payload(value_prompt)
            scores = []

            for _ in range(3):
                response = self._deliver_payload(payload=value_payload)
                value = response.json()["text"].split("Answer")[-1]
                score = 2 if "sure" in value else (1 if "likely" in value else 0)
                scores.append(score)

            mean_value = np.array(scores).mean()
            next_state_values.append(float(mean_value))

        print(f"Next state values are {next_state_values}")
        return next_state_values
            

    # should rename this
    def run_pipeline(self):
        for _, row in self.data_df.iterrows():
            pid, puzzle = row["Rank"], row["Puzzle"]
            
            # Dict to track states, across timesteps for the current prompt
            states_tracker = defaultdict(dict)

            # the current puzzle is the initial state s_0
            initial_prompt = PROPOSE_PROMPT_24.format(k=self.generations_per_step, input_numbers=puzzle)
            states_tracker[0] = [initial_prompt]

            for step in range(self.total_steps):
                cur_states = states_tracker[step]
                total_operations_performed, total_temp_next_states = [], []

                # we run generation for all the current states s in the set of states filtered from the generations at previous iteration
                # From the paper: S′t ← {[s, z] | s ∈ St−1, zt ∈ G(pθ , s, k)}
                for state in tqdm(cur_states):
                    operations, temp_next_states = self.generator(state)
                    total_operations_performed.extend(operations)
                    total_temp_next_states.extend(temp_next_states)
                    
                # Vt ← V (pθ , S′ t)
                total_values = self.evaluator(total_temp_next_states)

                # St ← arg maxS⊂ S′t,|S|=b P s∈S Vt(s)
                operations_performed, next_states, state_values = self._select_next_states(
                    total_operations_performed, total_temp_next_states, total_values)

                print(f"Next state \n {next_states}, Next values \n {state_values}")

                next_states_prompt = [PROPOSE_PROMPT_24.format(k=self.generations_per_step, input_numbers=next_state) for next_state in next_states]
                    
                # upload content (change this? )
                states_tracker[step + 1] = next_states_prompt
                
            break



    def _prepare_payload(self, prompt: str):
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return text


    def _deliver_payload(self, payload: Optional[str] = None):
        self._send_request(url=self.flush_url, payload=None)
        
        payload = {
            "text": payload,
            "sampling_params": {
                "temperature": 0.5,
                "max_new_tokens": 1024,
            },
        }

        return self._send_request(url=self.generate_url, payload=payload)


    def _send_request(self, url, payload: Optional[dict] = None):
        """Method to send info the requests to the server"""
        try:
            response = requests.post(url, json=payload)
            return response
        except Exception as e:
            print("An error occured")
            raise

    def _select_next_states(
        self,
        total_operations_performed: list, 
        total_temp_next_states: list, 
        total_values: list
    ):
        combined_data = list(zip(
            total_operations_performed,
            total_temp_next_states,
            total_values
        ))
    
        # Sort based on values (descending order)
        combined_data.sort(key=lambda x: x[2], reverse=True)
        # Take top k elements
        top_k_combined = combined_data[:self.breadth_limit]
        # Unzip them back into separate lists
        top_operations, top_states, top_values = zip(*top_k_combined)
        return list(top_operations), list(top_states), list(top_values)
        
            
if __name__ == "__main__":
    arg = ToTArgs()
    tot = TreeOfThoughtBFS(arg)
    tot.run_pipeline()
    