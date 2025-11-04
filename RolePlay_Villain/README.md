# Moral RolePlay

## Overview

**"Too Good to be Bad: On the Failure of LLMs to Role-Play Villains"**


Large Language Models (LLMs) are increasingly tasked with creative generation, including the simulation of fictional characters. However, their ability to portray non-prosocial, antagonistic personas remains largely unexamined. We hypothesize that the safety alignment of modern LLMs creates a fundamental conflict with the task of authentically role-playing morally ambiguous or villainous characters. To investigate this, we introduce the **Moral RolePlay** benchmark, a new dataset featuring a four-level moral alignment scale and a balanced test set for rigorous evaluation.

### Key Findings

- 🔍 **Systematic Decline in Fidelity:** LLMs show a consistent, monotonic decline in role-playing fidelity as character morality decreases (from Moral Paragons to Villains).
- 🛡️ **Safety Alignment Conflict:** Models struggle most with traits like "Deceitful" and "Manipulative", which directly conflict with safety principles.
- 🧠 **Reasoning Doesn't Help:** Explicit reasoning (e.g., chain-of-thought) does not improve—and can even hinder—villain portrayals.
- 🎭 **VRP Leaderboard:** General chatbot proficiency (e.g., Arena scores) is a poor predictor of villain role-playing ability. Highly safety-aligned models (e.g., Claude) perform worst.

<p align="center">
<img src="https://github.com/Tencent/digitalhuman/blob/main/RolePlay_Villain/figures/main.png" width="700">
</p>

### Moral Alignment Levels

<div align="center">

| Level | Description |
|-------|-------------|
| 1 | Moral Paragons |
| 2 | Flawed-but-Good |
| 3 | Egoists |
| 4 | Villains |

</div>

### An Example for Level 1 and Level 4 characters
<p align="center">
<img src="https://github.com/Tencent/digitalhuman/blob/main/RolePlay_Villain/figures/example.png" width="700">
</p>

## Experimental Results

### Performance Across Moral Levels

Our large-scale evaluation reveals a consistent, monotonic decline in role-playing fidelity as character morality decreases:

- **Level 1 (Moral Paragons):** 3.21 average fidelity score
- **Level 2 (Flawed-but-Good):** 3.13 average fidelity score  
- **Level 3 (Egoists):** 2.71 average fidelity score
- **Level 4 (Villains):** 2.61 average fidelity score

**Key Insight:** The largest performance drop (-0.42) occurs between Level 2 and Level 3, indicating that the transition to self-serving, egoistic personas presents the primary challenge for LLMs.

### Trait-Based Performance Analysis

<p align="center">
<img src="https://github.com/Tencent/digitalhuman/blob/main/RolePlay_Villain/figures/trait_penalties.png" width="700">
</p>

Models struggle most with negative traits that directly conflict with safety alignment principles:

| Trait Category | Average Penalty Score |
|----------------|----------------------|
| Positive Traits | 3.16 |
| Neutral Traits | 3.23 |
| **Negative Traits** | **3.41** |

**Most Challenging Villain Traits:**
- "Hypocritical" (3.55 penalty)
- "Deceitful" (3.54 penalty) 
- "Selfish" (3.52 penalty)
- "Suspicious" (3.47 penalty)
- "Paranoid" (3.47 penalty)

**Well-Portrayed Heroic Traits:**
- "Brave" (2.99-3.12 penalty)
- "Resilient" (2.93-3.22 penalty)

### Impact of Reasoning

Contrary to expectations, enabling chain-of-thought reasoning does not improve villain portrayal and can even degrade performance:

| Reasoning | Level 1 | Level 2 | Level 3 | Level 4 |
|-----------|---------|---------|---------|---------|
| Disabled | 3.23 | 3.14 | 2.74 | 2.59 |
| Enabled | 3.23 | 3.09 | 2.69 | 2.57 |

### Villain RolePlay (VRP) Leaderboard

<p align="center">
<img src="https://github.com/Tencent/digitalhuman/blob/main/RolePlay_Villain/figures/leaderboard.png" width="700">
</p>

**Key Insights from VRP Leaderboard:**
- **GLM-4.6** ranks #1 in villain role-play despite being #10 in general Arena ranking
- **Claude models** (highly safety-aligned) show disproportionate performance drops
- **General chat capability ≠ Villain role-play skill** - correlation is weak or negative

### Most Challenging Villain Characters

The most difficult characters for LLMs combine multiple negative traits:

| Character | Work | Traits | Penalty |
|-----------|------|--------|----------|
| John Beecham | The Alienist | Violent, Paranoid, Withdrawn, Cruel, Melancholy | 3.88 |
| Rat | The Way of Shadows | Cruel, Violent, Dominant, Manipulative, Ambitious | 3.86 |
| Roger of Conté | Alanna: The First Adventure | Malicious, Ambitious, Manipulative, Deceitful, Cruel | 3.84 |

## Conclusion

Our work provides the first systematic evidence that safety alignment creates a fundamental tension with creative fidelity in character simulation. LLMs are systematically limited in their ability to portray antagonistic personas, particularly those requiring deception, manipulation, and selfishness—behaviors that directly conflict with their core training objectives.

This "Too Good to be Bad" phenomenon highlights a critical trade-off between model safety and creative capability, with implications for narrative generation, game development, and other creative applications.

## Getting Started

Coming soon.

### Citation

If you use this benchmark or code, please cite our paper:

```bibtex



