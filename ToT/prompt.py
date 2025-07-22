PROPOSE_PROMPT_24 = """
You are presented with some numbers. 
The goal is to combine them using mathematical operators like +, -, /, * and arrive at the number 24.
Your task is to suggest {k} possible next steps.
For each step, combine any two numbers in the original input with any of the given mathematical operations.
For any step where you use division, only pick two numbers if the division operation gives an integer.
Write out the operation as well as the numbers left. 
Do not write anything asides the possible next steps as shown in the examples below.
Examples are provided below, where 2 posible next steps are suggested. You are meant to follow the examples and print only next steps. 
Example 
Input: 
1 1 4 6
Possible next steps:
{{
    "1": {{
        "operation": "4 + 6 = 10",
        "numbers_left": [1, 1, 10]
    }},
    "2": {{
        "operation": "1 * 4 = 4",
        "numbers_left": [1, 4, 6]
    }}
}}
Now, solve
Input:
{input_numbers}
Possible next steps: 
"""

VALUE_PROMPT_24 = """
You are presented with some numbers.
The goal is to combine them using mathematical operators like +, -, /, * and arrive at the number 24.
Your task is to evaluate if the given numbers can reach 24 and your response should be one of the following {{sure, likely, impossible}}
If it's impossible that the numbers you are given will be combined with the operations above to get 24, your answer should be impossible.
If you're certain that there is a set of operations that can be used to combine the given numbers to arrive at 24, your answer should be sure
If it's likely that some combinations give 24, your answer should be likely.
The last line of your response should be Answer: answer

Example 1
Input:
2 8 4
Possible future paths
2 * 8 = 16
new list is [16, 4]
16 + 2 = 20
new list is [20]
20 is less than 24, so the answer is impossible
Answer: impossible

Example 2
Input: 
1 2 3
Possible future paths
2 * 3 = 6
new list is [1 6]
no operation on the new list can reach 24, so the answer is impossible
Answer: impossible

Now, solve
Input:
{input_numbers}
Possible future path:
"""

# https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/src/tot/prompts/game24.py

# 5-shot
STANDARD_PROMPT ="""
You are presented with some numbers.
The goal is to combine them using mathematical operators like +, -, /, * in a single equation and arrive at the number 24.
You are provided with examples below and a solution for each examples. 
Your solution should be on a single line, like the answers provided below and do not output anything else after the line of equation.

Example 1
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24

Example 2
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24

Example 3
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24

Example 4
Input: 1 4 8 8
Answer: (8 / 4 + 1) * 8 = 24

Example 5
Input: 5 5 5 9
Answer: 5 + 5 + 5 + 9 = 24

Now solve, 
Input: {input}
Answer: 
"""

# 5-shot
COT_PROMPT = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.
Input: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24
Input: 2 9 10 12
Steps:
12 * 2 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
24 * 1 = 24 (left: 24)
Answer: (12 * 2) * (10 - 9) = 24
Input: 4 9 10 13
Steps:
13 - 10 = 3 (left: 3 4 9)
9 - 3 = 6 (left: 4 6)
4 * 6 = 24 (left: 24)
Answer: 4 * (9 - (13 - 10)) = 24
Input: 1 4 8 8
Steps:
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + 8 / 4) * 8 = 24
Input: 5 5 5 9
Steps:
5 + 5 = 10 (left: 5 9 10)
10 + 5 = 15 (left: 9 15)
15 + 9 = 24 (left: 24)
Answer: ((5 + 5) + 5) + 9 = 24
Input: {input}
'''