import os
import csv
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
from typing import Dict, List, Optional, Tuple


from prompt import PROPOSE_PROMPT_24, VALUE_PROMPT_24

@dataclass
class ToTArgs:
    generations_per_step: int = 8 # Number of thoughts to generate per step
    total_steps: int = 3 # Max tree dedpth
    data_dir: str = "data/game_of_24_easy.csv" # Path to game of 24 dataset
    model_name: str = "Qwen/Qwen2.5-32B" # LLM to use for gen and eval
    num_eval_attempts: int = 3 # Number of evaluation attempts per state
    max_gen_attempts: int = 10 # Max attempts for generation if API fails
    cache_file: str = "results_cache.json" # File to cache results

    # sglang-rollout specific args
    temperature: float = 0.5  # Temperature for sampling
    max_new_tokens: int = 2048 # Max number of tokens to generate

    # BFS-specific arg
    breadth_limit: int = 1 # Number of top states to keep at each step


@dataclass
class State:
    """
    Represents a state in the search tree containing:
    - numbers: Current numbers remaining
    - operation_history: Sequence of operations taken to reach this state
    - parent: Reference to parent state
    """
    numbers: List[int]
    operation_history: List[str]
    parent: Optional['State'] = None

    def to_prompt_string(self) -> str:
        """Convert numbers to a space-separated string for prompt generation."""
        return " ".join(map(str, self.numbers))


class TreeOfThought(ABC):
    """Abstract base class for ToT implementations"""
    def __init__(self, args: ToTArgs):
        pass
    def generator(self): 
        """Generate potential next states from current state"""
        pass
    def evaluator(self):
        """Evaluate quality of generated states"""
        pass


class TreeOfThoughtBFS(TreeOfThought):
    """Concrete implementation of Tree of Thought using Breadth-First Search strategy"""
    
    def __init__(self, args: ToTArgs):
        """Initialize BFS-based Tree of Thought solver with configuration"""
        self.generations_per_step = args.generations_per_step
        self.total_steps = args.total_steps
        self.breadth_limit = args.breadth_limit
        self.model_name = args.model_name
        self.temperature = args.temperature
        self.max_new_tokens = args.max_new_tokens
        self.num_eval_attempts = args.num_eval_attempts
        self.max_gen_attempts = args.max_gen_attempts
        self.data_df = pd.read_csv(args.data_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.cache_file = args.cache_file

        # SGLANG API endpoints
        self.generate_url = "http://localhost:30000/generate"
        self.flush_url = "http://localhost:30000/flush_cache"

        # Initialize cache
        self.cache = self._load_cache()


    def _load_cache(self) -> dict:
        """Load cache from file if it exists, otherwise return empty dict"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self) -> None:
        """Save current cache to file"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
        
    def generator(self, state: State) -> Tuple[List[str], List[State]]:
        """
        Generate possible next steps from current state using LLM
        
        Args:
            state: Current state in the search tree
            
        Returns:
            Tuple of (operations, next_states) where:
            - operations: List of operation represented as strings
            - next_states: List of resulting State objects
        """
        
        prompt = PROPOSE_PROMPT_24.format(k=self.generations_per_step, input_numbers=state.to_prompt_string())
        payload = self._prepare_payload(prompt)

        for attempt in range(self.max_gen_attempts):  
            try:
                response = self._deliver_payload(payload=payload)
                next_state_data = json.loads(response.json()["text"])
                operations, next_states = [], []
    
                for key, value in next_state_data.items():
                    op, num_left = value["operation"], value["numbers_left"]
                    assert len(num_left) == len(state.numbers) - 1, f"There should be {len(state.numbers) - 1} numbers left"
    
                    new_state = State(
                        numbers=[int(n) for n in num_left], 
                        operation_history=state.operation_history + [op],
                        parent=state
                    )
                    operations.append(op)
                    next_states.append(new_state)
    
                return operations, next_states
    
            except Exception as e:
                if attempt < self.max_gen_attempts - 1:
                    print(f"Generator attempt {attempt+1} failed: {e}. Retrying...")
                else:
                    print(f"Generator failed after {self.max_gen_attempts} attempts: {e}")
                    return [], [] 
   
    def evaluator(self, temp_next_states: List) -> List[float]:
        """
        Evaluate the quality of generated states using LLM
        Map sure/likely/impossible to 2/1/0 and take the average over num_eval_attempts attempts
        
        Args:
            temp_next_states: List of states to evaluate
            
        Returns:
            List of average evaluation score for each state
        """
        next_state_values = []
        print("Evaluating states")
        for state in temp_next_states:
            prompt = VALUE_PROMPT_24.format(input_numbers=state.to_prompt_string())
            payload = self._prepare_payload(prompt)
            scores = []

            for _ in range(self.num_eval_attempts):
                try:
                    response = self._deliver_payload(payload=payload)
                    value = response.json().get("text", "").split("Answer:")[-1].strip().lower()
                    score = 2 if "sure" in value else (1 if "likely" in value else 0)
                    scores.append(score)
                except Exception as e:
                    print(f"Error evaluating state {state.numbers}: {e}")
                    scores.append(0) 

            mean_value = np.array(scores).mean()
            next_state_values.append(float(mean_value))

        print(f"Next state values are {next_state_values}")
        return next_state_values
            
    def run_tot(self):
        """
        Main pipeline to solve all puzzles in dataset using ToT approach
        """
        answer_found, idx = 0, 0
        for _, row in tqdm(self.data_df.iterrows(), desc=f"Processing {idx + 1} of 50 "):
            pid, puzzle = row["Rank"], row["Puzzle"]

            # Check if puzzle is in cache
            if puzzle in self.cache:
                print(f"Puzzle {pid} found in cache, skipping...")
                if self.cache[puzzle]:
                    answer_found += 1
                idx += 1
                continue

            print(puzzle)

            # Flush cache before processing a new puzzle
            self._send_request(url=self.flush_url, payload=None)
            
            # Dict to track states, across timesteps for the current prompt
            states_tracker = defaultdict(dict)
            solution_found = False
            solution_tracker = []

            # the current puzzle is the initial state s_0
            initial_numbers = [int(n) for n in puzzle.split()]
            initial_state = State(numbers=initial_numbers, operation_history=[])
            states_tracker[0] = [initial_state]

            for step in range(self.total_steps):
                cur_states = states_tracker[step]
                total_operations_performed, total_temp_next_states = [], []

                # we run generation for all the current states s in the set of states filtered from the generations at previous iteration
                # From the paper: S′t ← {[s, z] | s ∈ St−1, zt ∈ G(pθ , s, k)}
                for state in tqdm(cur_states, desc=f"Step {step + 1}"):
                    # print(state)
                    operations, temp_next_states = self.generator(state)
                    total_operations_performed.extend(operations)
                    total_temp_next_states.extend(temp_next_states)
                    
                # Vt ← V (pθ , S′ t)
                total_values = self.evaluator(total_temp_next_states)

                # Select top states
                # St ← arg maxS⊂ S′t,|S|=b P s∈S Vt(s)
                operations, next_states, state_values = self._select_next_states(
                    total_operations_performed, total_temp_next_states, total_values
                )

                print(f"Selected states: {[s.numbers for s in next_states]}")
                print(f"State values: {state_values}")

                # Store next states for the next step
                states_tracker[step + 1] = next_states

            # Check for solutions
            for state in states_tracker[self.total_steps]:
                if len(state.numbers) == 1 and state.numbers[0] == 24:
                    answer_found += 1
                    solution_found = True
                    solution_tracker.append(state)
                    print(f"Solution found: {state.operation_history}")
                    break

            # Update cache
            self.cache[puzzle] = solution_found
            self._save_cache()
                
            idx += 1

        print(f"{answer_found} / {len(self.data_df)} answers were found")

        accuracy = (answer_found / len(self.data_df)) * 100

        with open("results_easy.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([f"b={self.breadth_limit}", accuracy])



    def _prepare_payload(self, prompt: str) -> dict:
        """Prepare payload for API request"""
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {
            "text": text,
            "sampling_params": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
            },
        }


    def _deliver_payload(self, payload: dict) -> requests.Response:
        try:
            return self._send_request(url=self.generate_url, payload=payload)
        except Exception as e:
            print(f"Error sending request: {e}")
            raise
        
        return self._send_request(url=self.generate_url, payload=payload)


    def _send_request(self, url: str, payload: Optional[dict] = None) -> requests.Response:
        """Send requests to the server"""
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response
        except Exception as e:
            print(f"An error {e} occured while sending request")
            raise

    def _select_next_states(
        self,
        total_operations_performed: list, 
        total_temp_next_states: list, 
        total_values: list
    )-> Tuple[List[str], List[State], List[float]]:
        """
        Select top states based on evaluation scores
        
        Args:
            total_operations_performed: All generated operations
            total_temp_next_states: All generated states
            total_values: Evaluation scores for each state
            
        Returns:
            Tuple of (top operations, top states, top values) limited by breadth
        """
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
    tot.run_tot()
    