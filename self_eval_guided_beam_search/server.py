import requests
from typing import Optional, List, Union
from sglang.test.test_utils import is_in_ci

if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

class Server:
    def __init__(self, model_path: str, port: Optional[int] = 30000) -> None:
        self.server_process, self.port = launch_server_cmd(
            f"python3 -m sglang.launch_server --model-path {model_path} --port {port} --host 0.0.0.0 --tokenizer-path {model_path}  --log-level error --mem-fraction-static 0.9"
        )
        wait_for_server(f"http://localhost:{self.port}")

    @property
    def shutdown(self):
        """Terminates the server after all execution is completed"""
        terminate_process(self.server_process)

    def flush_server(self) -> None:
        """Flushes the server cache. Running this method before every inference."""
        url = f"http://localhost:{self.port}/flush_cache"
        self._send_request(url)


    def _send_request(self, url, data: Optional[dict] = None):
        """Method to send info the requests to the server"""
        try:
            response = requests.post(url, json=data)
            return response
        except Exception as e:
            self.shutdown
            raise

    def _prepare_payload(
        self, 
        prompt: Union[List[str], List[int]],
        num_generations: int, 
        stop_at_newline: bool, 
        eval_prompt: bool,
        sampling_temp: float,
        max_tokens: int,
    ):
        payload = {
            "input_ids": prompt,
            "return_logprob": True,
            "sampling_params": {
                "max_new_tokens": max_tokens,
                "repetition_penalty": 1.05,
                "top_p": 0.8,
                "top_k": 20,
            }
        }
        payload["sampling_params"]["temperature"] = sampling_temp
        if num_generations is not None:
            payload["sampling_params"]["n"] = num_generations
        if stop_at_newline:
            payload["sampling_params"]["stop"] = "/r"
        if eval_prompt:
            payload["return_text_in_logprobs"] = True
            payload["token_ids_logprob"] = [4346, 5349, 32, 33]
        return payload
        
    def inference(
        self, 
        prompt: Union[List[int], List[str]], 
        stop_at_newline: bool,
        eval_prompt: bool = False,
        num_generations: int = None,
        sampling_temp: float = 0.6,
        max_tokens: int = 2048
    ):
        try:
            self.flush_server()
            url = f"http://localhost:{self.port}/generate"
            payload = self._prepare_payload(
                prompt, 
                num_generations, 
                stop_at_newline, 
                eval_prompt,
                sampling_temp,
                max_tokens=max_tokens
            )
            return self._send_request(url, payload).json()
        except Exception as e:
            self.shutdown
            raise