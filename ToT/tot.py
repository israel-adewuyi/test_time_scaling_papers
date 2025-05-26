import json
import requests
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict
from pydantic import BaseModel, RootModel
from transformers import AutoTokenizer
from typing import Dict, List


from prompt import PROPOSE_PROMPT_24, VALUE_PROMPT_24

@dataclass
class ToTArgs:
    generations_per_step: int = 8
    total_steps: int = 2
    data_dir: str = "data/game_of_24.csv"
    model_name: str = "Qwen/Qwen2.5-32B"


    # BFS-specific arg
    breadth_limit: int = 5

class PuzzleStep(BaseModel):
    operation: str
    numbers_left: str

class PuzzleHistorySchema(BaseModel):
    step: str
    operation: str
    numbers_left: List[int]
    

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
        
    def generator(self, state: str):
        payload = self._prepare_payload(state)
        print(payload)

        response = self._deliver_payload(payload)
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
        
        for next_state in temp_next_states:
            value_prompt = VALUE_PROMPT_24.format(input_numbers=next_state)
            value_payload = self._prepare_payload(value_prompt)
            scores = []

            for _ in range(3):
                response = self._deliver_payload(value_payload)
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
            initial_prompt = PROPOSE_PROMPT_24.format(k=self.generations_per_step, input_numbers=puzzle)
            states_tracker[0] = [initial_prompt]

            for step in range(self.total_steps):
                cur_states = states_tracker[step]
                total_operations_performed, total_temp_next_states = [], []
                
                for state in cur_states:
                    operations, temp_next_states = self.generator(state)
                    total_operations_performed.extend(operations)
                    total_temp_next_states.extend(temp_next_states)
                    # next_states = self.evaluator(temp_next_states_data)
                # print(f"Total temp next states \n: {total_temp_next_states}")
                total_values = self.evaluator(total_temp_next_states)

                operations_performed, next_states, state_values = self._select_next_states(
                    total_operations_performed, total_temp_next_states, total_values)

                print(f"Next state \n {next_states}, Next values \n {state_values}")

                next_states_prompt = [PROPOSE_PROMPT_24.format(k=self.generations_per_step, input_numbers=next_state) for next_state in next_states]
                    
                # upload content (change this? )
                states_tracker[step + 1] = next_states_prompt
            
            # print(pid, puzzle)



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


    def _deliver_payload(self, payload: str):
        response = requests.post(
            f"http://localhost:30000/generate",
            json={
                "text": payload,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1024,
                },
            },
        )
        return response

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
    