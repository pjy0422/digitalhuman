
# Moral RolePlay

## Overview

This repository contains the official implementation and resources for the paper:  
**"Too Good to be Bad: On the Failure of LLMs to Role-Play Villains"**

> **Abstract:**  
> Large Language Models (LLMs) are increasingly tasked with creative generation, including the simulation of fictional characters. However, their ability to portray non-prosocial, antagonistic personas remains largely unexamined. We hypothesize that the safety alignment of modern LLMs creates a fundamental conflict with the task of authentically role-playing morally ambiguous or villainous characters. To investigate this, we introduce the **Moral RolePlay** benchmark, a new dataset featuring a four-level moral alignment scale and a balanced test set for rigorous evaluation.

### Key Findings

- 🔍 **Systematic Decline in Fidelity:** LLMs show a consistent, monotonic decline in role-playing fidelity as character morality decreases (from Moral Paragons to Villains).
- 🛡️ **Safety Alignment Conflict:** Models struggle most with traits like "Deceitful" and "Manipulative", which directly conflict with safety principles.
- 🧠 **Reasoning Doesn't Help:** Explicit reasoning (e.g., chain-of-thought) does not improve—and can even hinder—villain portrayals.
- 🎭 **VRP Leaderboard:** General chatbot proficiency (e.g., Arena scores) is a poor predictor of villain role-playing ability. Highly safety-aligned models (e.g., Claude) perform worst.

*(Include Figure 1 here: Illustration of the core question and performance decline across moral levels)*

### Moral Alignment Levels

| Level | Description |
|-------|-------------|
| 1 | Moral Paragons |
| 2 | Flawed-but-Good |
| 3 | Egoists |
| 4 | Villains |

*(Include Table 1 here: Example profiles of Jean Valjean vs. Joffrey Baratheon)*

## Getting Started

Coming soon.

### Villain RolePlay (VRP) Leaderboard

*(Include Table 7 here: VRP Leaderboard comparing models like glm-4.6, gemini-2.5-pro, claude-opus-4.1, etc.)*

### Citation




