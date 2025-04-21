from torch import Tensor
from typing import Dict
from transformers import AutoModelForCausalLM, AutoTokenizer

class SEvRArgs:
    # model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    device: str = "cuda:2"
    
class SEvR:
    def __init__(self, args: SEvRArgs) -> None:
        self.model_name = args.model_name
        self.device = args.device
        self.args = args

        self.__post_init__()

    def __post_init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print('Model has been loaded')

    def _tokenize_inputs(self, prompt: str) -> Dict: # Is the return type really tensor? verify!!
        input_toks = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        return input_toks
        

    def generate(self, prompt: str) -> None:
        input_toks = self._tokenize_inputs(prompt)
        outputs = self.model.generate(
            input_ids=input_toks["input_ids"],
            attention_mask=input_toks["attention_mask"],
            max_new_tokens=512,
            output_logits=True,
            return_dict_in_generate=True,
            num_return_sequences=4,
            do_sample=True,
            stop_strings=["."], 
            tokenizer=self.tokenizer,
        )
        ## Some stats.
        # print(input_toks["input_ids"].shape)
        # print(type(outputs.logits), len(outputs.logits))
        # print("Logits shape:", outputs.logits[0].shape)  # (batch_size, max_new_tokens, vocab_size)
        # print("Generated tokens shape:", outputs.sequences.shape)  # (batch_size, input_length + max_new_tokens)
        # print("First generated token's logits:", outputs.logits[0, 0, :])
        # print(outputs.logits.shape)
        # print(outputs[0].shape)
        # print(outputs.logits)



        # Decode and print generations
        input_length = input_toks["input_ids"].shape[1]  # Length of input prompt in tokens
        for i, sequence in enumerate(outputs.sequences):
            # Skip input tokens to show only generated text
            generated_tokens = sequence[input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(f"Generation {i+1}:\n{generated_text}\n{'='*40}")

    def evaluate(self, ):
        pass


if __name__ == "__main__":
    args = SEvRArgs()
    self_eval_reasoner = SEvR(args)
    example_prompt = "Marilyn's 1st record sold 10 times as many copies as Harald's. If they sold 88,000 copies combined, how many copies did Harald sell?"
    self_eval_reasoner.generate(example_prompt)
    
        
    