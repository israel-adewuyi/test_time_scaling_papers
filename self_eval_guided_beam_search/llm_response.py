# import torch
from typing import Dict, List
from torch import Tensor

class Response:
    def __init__(self, response: Dict, query_logprobs: Tensor, query_tokens: List) -> None:
        self.response_tokens = response["output_ids"]
        self.num_tokens = response["meta_info"]["completion_tokens"]
        self.output_token_logprobs = response["meta_info"]["output_token_logprobs"]
        self.query_logprobs = query_logprobs
        self.query_tokens = query_tokens

        self.__post_init__()

    def __post_init__(self) -> None:
        self.response_logprob_sum = self.query_logprobs
        for tok_info in self.output_token_logprobs:
            self.response_logprob_sum += tok_info[0]

        self.tokens = self.query_tokens + self.response_tokens
        
    @property
    def logprobs_sum(self) -> Tensor:
        return self.response_logprob_sum

    @property
    def get_tokens(self) -> List:
        return self.tokens