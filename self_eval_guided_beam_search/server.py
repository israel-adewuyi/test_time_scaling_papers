import requests
from typing import Optional, List
from sglang.test.test_utils import is_in_ci

if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

class Server:
    def __init__(self, model_path: str, port: Optional[int] = 30000) -> None:
        self.server_process, self.port = launch_server_cmd(
            f"python3 -m sglang.launch_server --model-path {model_path} --port {port} --host 0.0.0.0 --skip-tokenizer-init --log-level error --mem-fraction-static 0.4"
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

    def inference(self, prompt: List[int]):
        try:
            self.flush_server()
            url = f"http://localhost:{self.port}/generate"
            data = {
                "input_ids": prompt,
                "sampling_params": {
                        "max_new_tokens": 32
                    },
                "return_logprob":True,
            }
            return self._send_request(url, data).json()
        except Exception as e:
            self.shutdown
            raise