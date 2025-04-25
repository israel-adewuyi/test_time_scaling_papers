import os
from dotenv import load_dotenv
load_dotenv()

import re
import gc
import time
import wandb
import torch
import argparse
import matplotlib.pyplot as plt

from typing import List
from datasets import Dataset
from vllm import LLM, SamplingParams
from trl import SFTConfig, SFTTrainer
from arithmetic_dataset import generate_dataset

# load_dotenv()
print(os.environ.get('CUDA_VISIBLE_DEVICES'))  # Should output '1' [[1]][[2]][[3]]
print(f"Num devices is {torch.cuda.device_count()}")
# assert torch.cuda.device_count() == 10, torch.cuda.device_count()


class STaRTrainer:
    """
    An instantiation of a STaR (Self-Taught Reasoner) trainer for the symbolic reasoning task.
    See https://arxiv.org/pdf/2203.14465
    Handles the core logic of generating reasoning scratch-pad and finetuning on the scratchpad

    Args:
    model_path (str): Link to HF model
    num_digits (int): The max number of digit addition we are attempting
    num_samples (int): The number of samples to generate for each digit in range(1, num_digits + 1)
    num_iterations (int): The max number of iterations to run the reasoner for
    accuracy_per_iter (bool): If to plot the accuracy_plot after every iteration

    Attributes
    dataset: This is a generated dataset of M samples of i digit addition, for i = 1 to num_digits.
    correct_counts_per_iter: This counts the number of correct addition the model gets right, for each digit, for each iteration.
    self.total_counts_per_iter: This counts the total number of addition (num_samples) the model does, for each digit, for each iteration.
    sampling_params: default sampling params to be used with VLLM 
    system_prompt: default system prompt to be used for each dataset
    
    """
    def __init__(self, model_path: str, num_digits: int, num_samples:int, num_iterations: int, accuracy_per_iter: bool):
        self.original_model_path = model_path
        self.model_path = model_path
        self.num_digits = num_digits
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.accuracy_per_iter = accuracy_per_iter
        self.dataset = generate_dataset(self.num_digits, self.num_samples)
        
        self.correct_counts_per_iter = {i: [] for i in range(1, self.num_digits + 1)}
        self.total_counts_per_iter = {i: [] for i in range(1, self.num_digits + 1)}
        
        self.llm = None
        self.sampling_params = SamplingParams(
            temperature=0,
            top_p=1,
            max_tokens=1024,
            n=1,
            stop=["<QED>"]
        )
        
        self.SYSTEM_PROMPT = """
        A conversation between User and Assistant. The User provides two numbers for the Assistant to add. The Assistant should solve the addition step by step, enclosed in the <scratch> tag. After the closing scratch tag, write the final answer on a newline and <QED> on another newline.
        Example 1:
        User:
        6 2 4 + 2 5 9
        Assistant:
        <scratch>
        6 2 4 + 2 5 9 , C: 0
        2 + 5 , 3 C: 1
        6 + 2 , 8 3 C: 0
        , 8 8 3 C: 0
        0 8 8 3
        </scratch>
        8 8 3
        <QED>

        Example 2:
        User:
        2 9 + 5 7
        Assistant:
        <scratch>
        2 9 + 5 7 , C: 0
        2 + 5 , 6 C: 1 # added 9 + 7 = 6 carry 1
        , 8 6 C: 0 # added 2 + 5 + 1 = 8 carry 0
        0 8 6
        </scratch>
        8 6 
        <QED>
        
        \nUser: {question}\nAssistant:
        """

    def load_model(self):
        torch.cuda.empty_cache()
        print(f"{'-' * 20} Loading from {self.model_path}")
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=1,
            max_model_len=1024,
            seed=42,
            gpu_memory_utilization=0.4,
            enforce_eager=True,
        )

    def string_match(self, input_text):
        pattern = r"\n*\s*<scratch>[\s\S]*?<\/scratch>\n\s*([\d\s]+)"
        match = re.fullmatch(pattern, input_text)
        if match:
            digits_part = match.group(1).strip()
            if digits_part:
                return True, digits_part
            return False, None
        return False, None
        
    def inference(self, texts: List[str], answers: List[str], digits: List[str], iter: int):
        prompts = [self.SYSTEM_PROMPT.format(question=text) for text in texts]
        
        with torch.no_grad():
            outputs = self.llm.generate(prompts, self.sampling_params)
        
        self.filter_outputs(prompts, outputs, answers, digits, iter)
            
    def filter_outputs(self, texts, outputs, answers, digits, iter):
        filtered_dataset = []
        
        iter_correct = {i: 0 for i in range(1, self.num_digits + 1)}
        iter_total = {i: 0 for i in range(1, self.num_digits + 1)}

        for question, output, answer, digit in zip(texts, outputs, answers, digits):
            generated_text = [out.text for out in output.outputs]
            
            assert len(generated_text) == 1, f"Length of generated text should be 1, instead, it is {len(generated_text)}"
            assert isinstance(digit, int)
            
            iter_total[digit] += 1
            flag, digits_result = self.string_match(generated_text[0])
            
            if flag and digits_result == answer:
                iter_correct[digit] += 1
                filtered_dataset.append(question + generated_text[0])
                
        for d in range(1, self.num_digits + 1):
            self.correct_counts_per_iter[d].append(iter_correct[d])
            self.total_counts_per_iter[d].append(iter_total[d])

        print(self.correct_counts_per_iter)
        print(self.total_counts_per_iter)
        print(f"Filtered dataset size: {len(filtered_dataset)}")
        self.data = filtered_dataset

    def finetune(self):
        if self.data:
            dataframe = {"text": self.data}
            dataset = Dataset.from_dict(dataframe)
            sfttraining_args = SFTConfig(
                per_device_train_batch_size=8,
                gradient_accumulation_steps=4,
                max_seq_length=1024,
                output_dir="/tmp",
                run_name=self.model_path
            )
            
            trainer = SFTTrainer(
                self.original_model_path,
                train_dataset=dataset,
                args=sfttraining_args,
            )
            trainer.train()
            output_path = f"fted_model_iter_{int(time.time())}"
            trainer.save_model(output_path)
            
            del trainer
            gc.collect()
            wandb.finish()
            torch.cuda.empty_cache()
            
            self.model_path = output_path

    def run(self):
        """
        This function contains the execution of the core logic itself.

        For each iteration, we do 
        1. inference, which consists of generating the scratchpads and filtering them for correctness
        2. Finetuning on the filtered dataset of correct scratchpads. 

        Note that for every finetuning step, we use the original model (load from HF afresh) but do inference on the finetuned model
        """
        
        texts = self.dataset["question"].tolist()
        answers = self.dataset["answer"].tolist()
        digits = self.dataset['digit'].tolist()
        print(f"Initial dataset size: {len(texts)}")
        
        for itr in range(self.num_iterations):
            print(f"Starting iteration {itr + 1}")
            
            gc.collect()
            torch.cuda.empty_cache()

            try:
                self.load_model()
                self.inference(texts, answers, digits, itr)
            except Exception as e:
                print(f"Error during inference in iteration {itr + 1}: {str(e)}")
                self.plot_accuracy(itr, f"assets/{self.original_model_path.split('/')[-1]}_accuracy_plot_error_{itr}.png")
                raise
            
            if self.accuracy_per_iter:
                self.plot_accuracy(itr, f"assets/{self.original_model_path.split('/')[-1]}_accuracy_plot_{itr}.png")

            del self.llm
            gc.collect()
            torch.cuda.empty_cache()
                
            try:
                self.finetune()
            except Exception as e:
                print(f"Error during finetuning in iteration {itr + 1}: {str(e)}")
                self.plot_accuracy(itr, f"assets/{self.original_model_path.split('/')[-1]}_accuracy_plot_error_{itr}.png")
                raise
        
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        
        if not self.accuracy_per_iter:
            self.plot_accuracy(self.num_iterations - 1)
        
    def plot_accuracy(self, iter: int, file_name='assets/accuracy_plot.png'):
        iterations = range(iter + 1)  
        accuracy = {i: [] for i in range(1, self.num_digits + 1)}
    
        for iter_idx in range(iter + 1):
            for digits in range(1, self.num_digits + 1):
                acc = (self.correct_counts_per_iter[digits][iter_idx] / 
                       self.total_counts_per_iter[digits][iter_idx]) * 100
                accuracy[digits].append(acc)
    
        print("Accuracy per digit length per iteration:")
        for digits in range(1, self.num_digits + 1):
            print(f"Digit {digits}: {accuracy[digits]}")
    
        plt.figure(figsize=(12, 6))
        for digits in range(1, self.num_digits + 1):
            plt.plot(iterations, accuracy[digits], "o-", label=f"{digits} digits", markersize=6)
    
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.title(f"Accuracy per Digit Length for {self.original_model_path.split('/')[-1]}", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.7) 
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="STaRTrainer for arithmetic addition model training")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Path to the pre-trained model")
    parser.add_argument("--num_digits", type=int, default=5,
                        help="Maximum number of digits for the arithmetic problems")
    parser.add_argument("--num_samples", type=int, default=500, 
                       help="Number of samples to generate in the dataset for each digit addition")
    parser.add_argument("--num_iterations", type=int, default=20,
                        help="Number of training iterations")
    parser.add_argument("--accuracy_per_iter", type=bool, default=True, 
                       help="Plot the graph of accuracy for each iterations")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    trainer = STaRTrainer(args.model_path, args.num_digits, args.num_samples, args.num_iterations, args.accuracy_per_iter)
    trainer.run()