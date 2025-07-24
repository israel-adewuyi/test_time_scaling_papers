# Implementation of [Tree of Thought](https://arxiv.org/abs/2305.10601) for Game of 24

## Implementation Details
- This repo implmements the [Tree of Thought Implementation](https://arxiv.org/abs/2305.10601) for the Game of 24, described in the paper.
- `scrape.py` scrapes the [4nums.com](https://www.4nums.com/game/difficulties/) website and save a subset of them to be used.
- `tot.py` contains the main implementation for ToT.
- `baseline.py` implements Input/Output (IO) and CoT prompting as baselines for the Game of 24 task (although I only ran experiments for IO due to computing resources).
- `prompt.py` contains all the prompts used across ToT task and the baseline tasks. The baseline prompts were copied from the original implementation of the paper.
- `data/` contains the actual subset of the Game of 24 I ran the experiments on.
- I ran ToT pipeline with two values of b, `b=1` and `b=5`.
- For the baseline, I ran IO task, with 100 iterations for a single game and averaging the performance.
- [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) was used for this experiments.
- All the experiments were conducted on a single Nvidia A100 GPU.

## Results.
| Approach | Results  | 
|----------|----------|
| IO       | 12%      | 
| ToT(b=1) | 72%      |
| ToT(b=5) | 92%      |

  
