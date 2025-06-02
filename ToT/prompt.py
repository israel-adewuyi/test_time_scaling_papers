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