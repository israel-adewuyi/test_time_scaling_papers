import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Transformer_Sampler:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
        self.model = model
        self.config = model.config
        print(self.config)
        self.tokenizer = tokenizer


if __name__ == "__main__":
    model_path = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    transformer_sampler = Transformer_Sampler(model, tokenizer)