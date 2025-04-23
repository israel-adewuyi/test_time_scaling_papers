import torch

from torch import Tensor
from jaxtyping import Float, Int
from transformers import AutoModelForCausalLM, AutoTokenizer

class Transformer_Sampler:
    """
    A class for generating text using a transformer-based causal language model.

    Attributes:
        model (AutoModelForCausalLM): The pre-trained transformer model for text generation.
        config: The configuration of the model.
        tokenizer (AutoTokenizer): The tokenizer corresponding to the model.
    """
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
        self.model = model
        self.config = model.config
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def sample(self, prompt: str, max_new_tokens: int, **kwargs) -> str:
        """
        Generates text by sampling tokens sequentially from the model.
        Args:
            prompt (str): The input text to start generation.
            max_new_tokens (int): Maximum number of new tokens to generate.
            **kwargs: Additional arguments for sampling (e.g., temperature).
        Returns:
            str: The generated text including the prompt.
        """
        input_toks = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        
        for _ in range(max_new_tokens):
            output = self.model(input_toks)
            logits = output.logits[0, -1, :]
            new_token = torch.tensor([Transformer_Sampler.sample_next_token(logits, **kwargs)], dtype=torch.int64)
            input_toks = torch.cat([input_toks, new_token[None,]], dim=-1)

        return self.tokenizer.decode(input_toks[0])
        
    @staticmethod
    def sample_next_token(
        logits: Float[Tensor, "d_vocab"],
        temperature: Float,
    ):
        """
        Samples the next token based on logits and temperature.
        Args:
            logits (Tensor): Logits output by the model for the last token.
            temperature (float): Sampling temperature. If 0, uses greedy search.
        Returns:
            int: The index of the sampled token.
        """
        if temperature == 0.0:
            return Transformer_Sampler.greedy_search(logits)

    
    @staticmethod
    def greedy_search(logits: Float[Tensor, "d_vocab"]):
        """
        Selects the token with the highest logit value (deterministic).
        Args:
            logits (Tensor): Logits output by the model for the last token.
        Returns:
            int: The index of the token with the highest logit.
        """
        return torch.argmax(logits).item()
        

    def __str__(self):
        """Returns string representation of the sampler"""
        return f"TransformerSampler for {self.config.model_type}"


if __name__ == "__main__":
    model_path = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.config.use_cache=False
    
    transformer_sampler = Transformer_Sampler(model, tokenizer)

    prompt = "The dog"
    print(transformer_sampler.sample(prompt, max_new_tokens=32, temperature=0.0))
