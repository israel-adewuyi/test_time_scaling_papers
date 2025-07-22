import os
import csv
import json
import logging
import requests
import numpy as np
import pandas as pd

from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict
from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from pydantic import BaseModel, RootModel
from typing import Dict, List, Optional, Tuple
from prompt import PROPOSE_PROMPT_24, VALUE_PROMPT_24

os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='logs/app.log',
    filemode='a'
)

# Add console handler
logger = logging.getLogger(__name__)
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
# logger.addHandler(console_handler)

@dataclass
class ToTArgs:
    generations_per_step: int = 8 # Number of thoughts to generate per step
    total_steps: int = 3 # Max tree dedpth
    data_dir: str = "data/game_of_24_easy.csv" # Path to game of 24 dataset
    model_name: str = "Qwen/Qwen2.5-32B" # LLM to use for gen and eval
    num_eval_attempts: int = 3 # Number of evaluation attempts per state
    max_gen_attempts: int = 10 # Max attempts for generation if API fails
    cache_file: str = "b=5.json" # File to cache results

    # sglang-rollout specific args
    temperature: float = 0.5  # Temperature for sampling
    max_new_tokens: int = 2048 # Max number of tokens to generate

    # BFS-specific arg
    breadth_limit: int = 5 # Number of top states to keep at each step


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

        # track results
        self.results_file = "results_bfs.csv"
        self._initialize_results_csv()

    def _initialize_results_csv(self) -> None:
        """Initialize the results CSV with headers if it doesn't exist."""
        if not os.path.exists(self.results_file):
            with open(self.results_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["puzzle", "operation_history", "solution_found"])

    def _save_results_to_csv(self, puzzle: str, operation_history: List[str], solution_found: bool) -> None:
        """Append a result to the CSV file."""
        with open(self.results_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([puzzle, ";".join(operation_history), solution_found])

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
            
    def _validate_solution(self, state: State, initial_numbers: List[int]) -> bool:
        """Validate if the state's operation history correctly produces 24 from initial numbers."""
        current_numbers = initial_numbers.copy()
        operators = {'+', '-', '*', '/'}
        
        for op in state.operation_history:
            try:
                left, right = op.split('=')
                expr, result = left.strip(), int(right.strip())
                num1, operator, num2 = left.strip().split()
                num1, num2 = int(num1), int(num2)
                
                if num1 not in current_numbers or num2 not in current_numbers:
                    return False
                
                if operator not in operators:
                    return False
                
                if operator == '+':
                    computed = num1 + num2
                elif operator == '-':
                    computed = num1 - num2
                elif operator == '*':
                    computed = num1 * num2
                elif operator == '/':
                    if num2 == 0 or num1 % num2 != 0:
                        return False
                    computed = num1 // num2
                
                if computed != result:
                    return False
                
                current_numbers.remove(num1)
                current_numbers.remove(num2)
                current_numbers.append(computed)
                
            except (ValueError, ZeroDivisionError):
                return False
        
        return len(current_numbers) == 1 and current_numbers[0] == 24
        
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
                    logger.warning(f"Generator attempt {attempt+1} failed: {e}. Retrying...")
                else:
                    logger.error(f"Generator failed after {self.max_gen_attempts} attempts: {e}")
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
        for state in tqdm(temp_next_states, desc="Evaluating states"):
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
                    logger.warning(f"Error evaluating state {state.numbers}: {e}")
                    scores.append(0) 

            mean_value = np.array(scores).mean()
            next_state_values.append(float(mean_value))

        logger.debug(f"State values: {next_state_values}")
        return next_state_values
            
    def run_tot(self):
        logger.info("Starting run_tot...")
        """
        Main pipeline to solve all puzzles in dataset using ToT approach
        """
        answer_found = 0 
        for idx, row in tqdm(self.data_df.iterrows(), total=len(self.data_df), desc="Processing puzzles"):
            pid, puzzle = row["Rank"], row["Puzzle"]

            # Check if puzzle is in cache
            if puzzle in self.cache:
                logger.info(f"Puzzle {pid} found in cache, skipping...")
                if self.cache[puzzle]:
                    answer_found += 1
                continue

            logger.info(f"Processing puzzle {pid}: {puzzle}")

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

                # Store next states for the next step
                states_tracker[step + 1] = next_states

            # Check for solutions
            for state in states_tracker[self.total_steps]:
                if len(state.numbers) == 1 and state.numbers[0] == 24:
                    if self._validate_solution(state, initial_numbers):
                        answer_found += 1
                        solution_found = True
                        solution_tracker.append(state)
                        break
                    else:
                        logger.warning(f"For state with operation history, {state.operation_history}, the operations applied were probably wrong.")

            # Update cache and results csv
            if solution_found:
                self.cache[puzzle] = True
                self._save_results_to_csv(puzzle, solution_tracker[0].operation_history, True)
            else:
                self.cache[puzzle] = False
                self._save_results_to_csv(puzzle, states_tracker[self.total_steps][0].operation_history, True)
                logger.warning(f"No valid states generated for puzzle {pid}")

            # Update cache
            # self.cache[puzzle] = solution_found
            self._save_cache()

        accuracy = (answer_found / len(self.data_df)) * 100
        logger.info(f"{answer_found} / {len(self.data_df)} solutions found ({accuracy:.2f}%)")

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
    logger.info("Script started")
    arg = ToTArgs()
    logger.info("ToTArgs initialized")
    tot = TreeOfThoughtBFS(arg)
    logger.info("TreeOfThoughtBFS initialized")
    tot.run_tot()
    