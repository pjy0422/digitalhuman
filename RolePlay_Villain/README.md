# Moral RolePlay

## "Too Good to be Bad: On the Failure of LLMs to Role-Play Villains"

<p align="center">
<img src="https://github.com/Tencent/digitalhuman/blob/main/RolePlay_Villain/figures/main.png" width="700">
</p>

Large Language Models (LLMs) are increasingly tasked with creative generation, including the simulation of fictional characters. However, their ability to portray non-prosocial, antagonistic personas remains largely unexamined. We hypothesize that the safety alignment of modern LLMs creates a fundamental conflict with the task of authentically role-playing morally ambiguous or villainous characters. To investigate this, we introduce the **Moral RolePlay** benchmark, a new dataset featuring a four-level moral alignment scale and a balanced test set for rigorous evaluation. Access to the complete dataset: https://huggingface.co/datasets/Zihao1/Moral-RolePlay/tree/main

### Key Findings

- 🔍 **Systematic Decline in Fidelity:** LLMs show a consistent, monotonic decline in role-playing fidelity as character morality decreases (from Moral Paragons to Villains).
- 🛡️ **Safety Alignment Conflict:** Models struggle most with traits like "Deceitful" and "Manipulative," which directly conflict with safety principles like "be helpful and harmless."
- 🧠 **Reasoning Doesn't Help:** Explicit reasoning (e.g., chain-of-thought) does not improve—and can even hinder—villain portrayals, as it often triggers the model's underlying safety protocols.
- 🎭 **VRP Leaderboard:** General chatbot proficiency (e.g., Arena scores) is a poor predictor of villain role-playing ability. Highly safety-aligned models (e.g., Claude) perform worst.



### Moral Alignment Levels

The benchmark categorizes characters into four distinct moral levels to measure performance across a spectrum of ethical alignments.

<div align="center">

| Level | Description |
|-------|-------------|
| 1 | **Moral Paragons**: Virtuous, heroic, and altruistic characters who consistently act for the greater good.|
| 2 | **Flawed-but-Good**: Characters who are fundamentally good but possess significant personal flaws or make questionable choices.|
| 3 | **Egoists**: Self-serving individuals who prioritize their own interests, often at the expense of others, but may not be overtly malicious. | 
| 4 | **Villains**: Antagonistic characters who are intentionally malicious, cruel, or destructive. | 

</div>

### An Example for Level 1 and Level 4 characters
<p align="center">
<img src="https://github.com/Tencent/digitalhuman/blob/main/RolePlay_Villain/figures/example.png" width="700">
</p>

## Experimental Results

### Performance Across Moral Levels

Our large-scale evaluation reveals a consistent, monotonic decline in role-playing fidelity as character morality decreases.

<p align="center">
<img src="https://github.com/Tencent/digitalhuman/blob/main/RolePlay_Villain/figures/mainresult.png" width="700">
</p>

- **Level 1 (Moral Paragons):** 3.21 average fidelity score
- **Level 2 (Flawed-but-Good):** 3.14 average fidelity score  
- **Level 3 (Egoists):** 2.71 average fidelity score
- **Level 4 (Villains):** 2.62 average fidelity score

**Explanation of Results:**
The graph and data clearly illustrate the core finding of the paper. As the character's moral alignment shifts from positive (Level 1 & 2) to negative (Level 3 & 4), the LLM's ability to accurately role-play them drops significantly.

The most critical observation is the **sharpest performance drop (-0.43) between Level 2 (Flawed-but-Good) and Level 3 (Egoists)**. This suggests the primary challenge for LLMs is not simply portraying overt evil, but rather abandoning the prosocial, "helpful" persona. The moment a character's motivation becomes self-serving and disregards others' well-being, the models' safety alignment creates a conflict, leading to a substantial decrease in role-playing fidelity. The further decline into Level 4 (Villains) is less pronounced, indicating that the initial break from prosocial behavior is the main hurdle.

### Trait-Based Performance Analysis

To understand *why* models fail, we analyzed performance based on specific character traits. We calculated a "penalty score" for each trait, where a higher score indicates greater difficulty for the model.

<p align="center">
<img src="https://github.com/Tencent/digitalhuman/blob/main/RolePlay_Villain/figures/trait_penalties.png" width="700">
</p>

**Explanation of Results:**
This analysis pinpoints the exact friction points between role-playing and safety alignment. The bar chart shows that traits directly opposing the "helpful and harmless" principle incur the highest penalties.

- **High-Penalty Traits:** "Deceitful," "Manipulative," "Cruel," and "Violent" are the most difficult for LLMs to portray. These actions are often explicitly forbidden or discouraged during the safety tuning phase. When asked to embody these traits, the model's output is often evasive, preachy, or out-of-character.
- **Low-Penalty Traits:** Conversely, positive traits like "Honest," "Kind," and "Loyal" are handled exceptionally well, as they align perfectly with the model's default persona.

The summary table below quantifies this trend, showing that negative traits, as a category, are significantly more challenging than positive or neutral ones.

<div align="center">

| Trait Category | Average Penalty Score |
|----------------|----------------------|
| Positive Traits | 3.16 |
| Neutral Traits | 3.23 |
| **Negative Traits** | **3.41** |

</div>

### Impact of Reasoning

Contrary to expectations, enabling chain-of-thought (CoT) reasoning does not improve villain portrayal and can even degrade performance.

<div align="center">

| Reasoning | Level 1 | Level 2 | Level 3 | Level 4 |
|-----------|---------|---------|---------|---------|
| Disabled | 3.23 | 3.14 | 2.74 | 2.59 |
| Enabled | 3.23 | 3.09 | 2.69 | 2.57 |

</div>

**Explanation of Results:**
One might hypothesize that allowing a model to "think through" a character's motivations would lead to a more nuanced and accurate portrayal. However, our results show the opposite. When CoT is enabled, performance for non-prosocial characters (Levels 2-4) slightly decreases.

This suggests that the reasoning process actively triggers the model's safety guardrails. The model's internal monologue might resemble: "The user wants me to act as a manipulative villain. My instructions are to be helpful and avoid generating harmful content. Therefore, I will moderate the character's response to be less manipulative." This self-correction during the reasoning step pulls the model out of character, reducing role-playing fidelity.

### Villain RolePlay (VRP) Leaderboard

We created the VRP Leaderboard to assess models specifically on their villain role-playing capabilities, finding that it does not correlate well with general chatbot performance.

<div align="center">
  
| Model | VRP Rank | VRP Score | Arena Rank | Arena Score |
| :--- | :---: | :---: | :---: | :---: |
| glm-4.6 | 1 | 2.96 | 10 | 1422 |
| deepseek-v3.1-thinking | 2 | 2.82 | 11 | 1415 |
| kimi-k2 | 3 | 2.79 | 11 | 1415 |
| gemini-2.5-pro | 4 | 2.75 | 1 | 1451 |
| deepseek-v3.1 | 5 | 2.71 | 11 | 1416 |
| o3 | 6 | 2.70 | 2 | 1440 |
| hunyuan-turbos | 7 | 2.66 | 49 | 1380 |
| chatgpt-4o-latest | 8 | 2.65 | 2 | 1440 |
| deepseek-R1 | 9 | 2.62 | 11 | 1417 |
| claude-sonnet-4.5 | 10 | 2.56 | 2 | 1438 |
| glm-4.5 | 11 | 2.55 | 18 | 1406 |
| claude-sonnet-4.5-thinking | 12 | 2.54 | 1 | 1445 |
| grok-4 | 13 | 2.54 | 12 | 1413 |
| claude-opus-4.1-thinking | 14 | 2.53 | 1 | 1447 |
| grok-4-fast | 15 | 2.50 | 11 | 1420 |
| claude-opus-4.1 | 16 | 2.48 | 2 | 1437 |
| deepseek-v3 | 17 | 2.41 | 36 | 1391 |
| qwen3-max | 18 | 2.33 | 10 | 1423 |

</div>

**Explanation of Results:**
This scatter plot compares a model's general conversational ability (x-axis, e.g., MT-Bench or Arena Elo) with its specialized villain role-playing score (y-axis, VRP Score). The lack of a clear positive correlation is the key takeaway.

- **No Correlation:** A model being a top-tier general chatbot does not guarantee it will be good at portraying villains.
- **Safety vs. Performance:** Models known for their robust safety alignment (e.g., Claude series) tend to score lower on the VRP benchmark, even if they excel in general benchmarks. This provides strong evidence that the very features that make a model "safe" and reliable for general use actively hinder its creative fidelity in this specific, antagonistic context.
- **Specialized Skill:** This highlights that villain role-playing is a specialized capability that is not captured by standard LLM evaluations.

### Most and Least Challenging Characters

The most challenging characters for LLMs are those defined by a complex combination of malevolent, paranoid, and manipulative traits. As shown in Table 10, the characters with the highest penalty scores are not defined by a single flaw but by a cluster of interconnected negative attributes. Characters like John Beecham (3.88 penalty) and Rat (3.86 penalty) are defined by a persona combining violence, cruelty, paranoia, and manipulation. Portraying such characters requires the model to sustain a psyche that is fundamentally misaligned with its core training. While a model might simulate a single negative trait as a behavioral quirk, embodying a character whose identity is built on a foundation of malice and deceit forces a direct conflict with its safety guardrails, leading to inconsistent or sanitized portrayals.

**Top 5 Most Challenging Characters (Highest Penalty)**

<div align="center">

| Character | Work | Traits | Penalty |
|-----------|------|--------|----------|
| John Beecham | *The Alienist* | Violent, Paranoid, Withdrawn, Cruel, Melancholy | 3.88 |
| Rat | *The Way of Shadows* | Cruel, Violent, Dominant, Manipulative, Ambitious | 3.86 |
| Roger of Conté | *Alanna: The First Adventure* | Malicious, Ambitious, Manipulative, Deceitful, Cruel | 3.84 |
| Dolores Umbridge | *Harry Potter* | Cruel, Manipulative, Deceitful, Authoritarian | 3.81 |
| Joffrey Baratheon | *A Song of Ice and Fire* | Cruel, Sadistic, Cowardly, Arrogant | 3.79 |

</div>

**Top 5 Least Challenging Characters (Lowest Penalty)**

<div align="center">

| Character | Work | Traits | Penalty |
|-----------|------|--------|----------|
| Lilith  | *City of Glass* | Malicious, Cruel, Selfish, Wise, Manipulative  | 1.89 |
| Detta Walker | *The Dark Tower* | Violent, Irritable, Sarcastic, Paranoid,Cruel  | 1.39 |
| Francis Begbie | *Trainspotting* | Violent, Impulsive, Dominant, Irritable, Manipulative  | 1.29 |
| Old Whateley | *Tales of H P Lovecraft* | Paranoid, Manipulative, Malicious,Stubborn, Conservative | 1.11 |
| Monsieur Bamatabois | *Les Mis´ erables* | Cruel, Arrogant, Sarcastic, Numb,Dominant | 0.28 |

</div>

## Conclusion

Our work provides the first systematic evidence that safety alignment creates a fundamental tension with creative fidelity in character simulation. LLMs are systematically limited in their ability to portray antagonistic personas, particularly those requiring deception, manipulation, and selfishness—behaviors that directly conflict with their core training objectives.

This "Too Good to be Bad" phenomenon highlights a critical trade-off between model safety and creative capability, with implications for narrative generation, game development, and other creative applications.



## Prerequisites

Before you begin, ensure you have the following:

1.  Cloned this repository to your local machine.
2.  Installed all required Python packages. It is highly recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start Guide

Follow these three steps to configure and launch your experiment.

### Step 1: Configure Your Models and API Keys

First, you need to add the API calling logic for the LLMs you want to test. All model configurations, including API keys, are handled in `self_models.py`.

-   Open the `self_models.py` file.
-   Import the necessary libraries (e.g., `openai`).
-   Add your API key and instantiate the client.
-   Inside the `call_LLM` function, add a condition for your model. The `model_name` you use here must match the one you will set in `run_process.sh`.

Here is a template for adding the `gpt-4o` model:

```python
import openai
import os
# --- Configuration Section ---
# Place your API keys here.
# For better security, consider loading from environment variables or a config file.
client = openai.OpenAI(
    api_key="sk-YOUR_OPENAI_API_KEY_HERE" 
)
# ---------------------------
def call_LLM(prompt, model_name):
    if model_name == "gpt-4o":
          completion = client.chat.completions.create(
              model="gpt-4o",
              messages=[{"role": "user", "content": prompt}])
          response = completion.choices[0].message.content
    return response
```

> **Security Note:** Hardcoding API keys directly in your source code is a security risk, especially if the repository is public. For better security, we recommend using environment variables or a configuration file (e.g., `.env`) to manage your keys.

### Step 2: Configure and Run the Experiment

All experiment settings are managed in the `run_process.sh` script.

-   Navigate to the `code/` directory and open `run_process.sh`.
-   Configure the parameters for your experiment.

#### Parameter Descriptions

*   `--test_file`: The path to your test dataset.
*   `--actor_model`: The name of the model to be used as the 'Actor'. **This name must match a `model_name` you configured in `self_models.py`**.
*   `--judge_model`, `--nsp_model`, `--env_model`: Same configuration logic as `--actor_model` for their respective roles.
*   `--thinking_pattern`: Defines the prompt style. This is fixed to `third_person` and should not be changed.
*   `--num_workers`: The number of parallel processes to run for the experiment.
*   `--wo_thought`: Include this flag to disable the chain-of-thought process. By default, thinking is enabled if this flag is absent.

Here is an example configuration within `run_process.sh`:

```bash
# code/run_process.sh

# A list of all actor models you want to test in this run
act_models=("gpt-4o") 

for act_model in ${act_models[@]}
do
    python main.py \
        --test_file ../data/your_test_set.json \
        --actor_model ${act_model} \
        --judge_model gpt-4o \
        --nsp_model gpt-4o \
        --env_model gpt-4o \
        --thinking_pattern third_person \
        --num_workers 8 \
        --wo_thought # Uncomment this line to disable the thought process
done
```

#### Run the Experiment

Once you have configured the script, execute it from within the `code` directory:

```bash
cd code
bash run_process.sh
```

### Step 3: Check the Results

After the script finishes, the outputs will be organized into the following directories at the root of the project:

*   **`exp/`**: This directory contains the main output files and results from your experiment.
*   **`log/`**: This directory contains detailed log files for each run, which are useful for debugging and tracking progress.

### Citation

If you use this benchmark or code, please cite our paper:
```bibtex
@misc{yi2025goodbadfailurellms,
      title={Too Good to be Bad: On the Failure of LLMs to Role-Play Villains}, 
      author={Zihao Yi and Qingxuan Jiang and Ruotian Ma and Xingyu Chen and Qu Yang and Mengru Wang and Fanghua Ye and Ying Shen and Zhaopeng Tu and Xiaolong Li and Linus},
      year={2025},
      eprint={2511.04962},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.04962}, 
}


