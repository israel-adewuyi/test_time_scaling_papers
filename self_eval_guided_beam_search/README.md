# Implementation of [Self-Evaluation Guided Beam Search for Reasoning](https://www.arxiv.org/pdf/2305.00633)

## Implementation details
### Inference Engine
[Sglang](https://docs.sglang.ai/index.html) was used as the inference engine. Default sampling parameters used across all experiments are
- `repetition_penalty` = 1.05
- `top_p` = 0.8
- `top_k` = 20
- `sampling_temperature` = 0.9

### Baseline
Chain-of-Thought (CoT) was implemented as a baseline with the following hyperparameters
- The default sglang sampling params were used
- `max_tokens` = 4096

### Experiments
| Experiment | num_samples (n) | num_beams (k) | tau (t) | alpha (a) | lambda (l) |
|  --------- | --------------- | ------------- | ------- | --------- | ---------- |
| Exp. 1     |       11        |      3        |   0.6   |    0.8    |     0.7    |
| Exp. 2     |       11        |      5        |   0.6   |    0.8    |     0.7    |

## Results
Model: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
| Experiment        | Result      |
|-------------------|-------------|
| CoT Baseline      | 86%         | 
| Exp 1             | 76%         |
| Exp 2             | 75%         | 
