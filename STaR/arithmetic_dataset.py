import numpy as np
import pandas as pd


def generate_random_numbers(n: int):
    low = 10 ** (n - 1)
    high = 10 ** n
    res = np.random.randint(low, high, size=2)
    return res

def format_with_spaces(num: int):
    return " ".join(str(num))
    
def generate_dataset(num_digits: int, num_samples: int):
    probs = [(1 / num_digits)] * num_digits
    question_list = []
    digit_list = []
    answer_list = []

    for idx in range(num_digits):
        for _ in range(num_samples):
            num1, num2 = generate_random_numbers(idx + 1)
            res = num1 + num2
    
            num1 = format_with_spaces(num1)
            num2 = format_with_spaces(num2)
            res = format_with_spaces(res)
    
            question_list.append(num1 + " + " + num2)
            answer_list.append(res)
            digit_list.append(idx + 1)

    assert len(question_list) == len(answer_list), "There should be a corresponding answer for every question"
    
    data = pd.DataFrame({
        "question": question_list,
        "answer": answer_list,
        "digit": digit_list
    })

    return data

# if __name__ == "__main__":
    
    # generate_random_numbers(3)
    # print(generate_dataset())