# Tree of Thought Implementation for Game of 24

This repository contains an implementation of the Tree of Thought (ToT) framework from the paper ["Tree of Thoughts: Deliberate Problem Solving with Large Language Models"](https://arxiv.org/abs/2305.10601) applied to the Game of 24.

## Overview

The Tree of Thought approach decomposes tasks into thoughts (decomposition is task-specific). Thoughts are generated as potential nodes on a tree and evaluation determines which thoughts are promising and are to be further explored. 

The goal of the Game of 24 is to combine four numbers using basic arithmetic operations (+,-,*,/) to reach the number 24.

In this implementation, BFS is used to evaluate and explore promising thoughts / states. 

`breadth_limit` promising thoughts / states are maintained per step.

I also implemented IO as baseline, with `100` iterations per puzzle.

All of these were evaluated with `Qwen/Qwen2.5-32B`

## Directory Structure
```
ToT/
├── data/                   # Game of 24 datasets scraped from [4nums.com](https://www.4nums.com/game/difficulties/)
├── tot.py                  # Main ToT implementation
├── scrape.py               # Web scraper for puzzles
├── prompts.py              # Prompt templates for both generation and evaluation
├── baseline.py             # Used to run baseline eval (I/O and CoT)
└── README.md
```           


## Results.
| Approach | Results  | 
|----------|----------|
| IO       | 12%      | 
| ToT(b=1) | 72%      |
| ToT(b=5) | 92%      |

  