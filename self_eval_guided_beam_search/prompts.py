SYSTEM_PROMPT = """
You are Qwen, created by Alibaba Cloud. You are a helpful assistant specialized in solving mathematical problems.

When presented with a math problem:
1. Reason through the problem step-by-step.
2. Show your full thinking process before giving the final answer.
3. Each reasoning step should be followed by "r"
3. Place the final answer inside \\boxed{}.

If you're continuing from prior steps, do not repeat previous reasoning—continue logically from the last point.
""".strip()

EVAL_SYSTEM_PROMPT = """
You are given a problem as well as an intermediate reasoning step
of an Assistant. Is the reasoning step (A) Correct (B) Incorrect?
""".strip()

FEW_SHOT_EXAMPLE_1_USER = """
Example 1 Question: Allison brought some CDs online. Each CD cost $7.
There was an additional charge of $4 per order for shipping costs.
The total bill came to $60. How many CDs did Allison buy?
Answer: Each CD cost 7 dollars. Is the above step of reasoning:
# (A) Correct # (B) Incorrect
""".strip()

FEW_SHOT_EXAMPLE_1_ASSISTANT = "The above step of reasoning is (A)"

FEW_SHOT_EXAMPLE_2_USER = """
Example 2 Question: Allison brought some CDs online. Each CD cost $7.
There was an additional charge of $4 per order for shipping costs.
The total bill came to $60. How many CDs did Allison buy?
Answer: Each CD cost 7 dollars. And there was an additional charge
of 4 dollars. So the total cost of each CD is 7 + 4 = 11 dollars.
Is the above step of reasoning: # (A) Correct # (B) Incorrect
""".strip()

FEW_SHOT_EXAMPLE_2_ASSISTANT = "The above step of reasoning is (B)"