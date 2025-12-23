import json
import os
from tqdm import tqdm
from typing import List, Dict
import sys
sys.path.append('../src')  
from base import Task, Agent, StepEfficientEnv, StepRandomEnv, StepRandomFairEnv, StepEfficientFairEnv, PosteriorOptimalEnv, IdealEnv, GreedyEnv
from utils import load_data, printf, DOCUMENTATIONS
from metric import *
import argparse
import numpy as np
from collections import defaultdict
import random
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multiple gpus
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def mean(arr):
    return sum(arr) / len(arr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='./log/efficient')
    parser.add_argument("--persona", type=str, default='efficient')
    parser.add_argument("--hp", type=int, default=11)
    parser.add_argument("--remove", action="store_true", default=False, help="Enable removal option")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--bar", type=float, default=0.4)
    parser.add_argument("--bad", type=int, default=4)
    parser.add_argument("--input_file", type=str,
                        default = 'batch_tasks.63.batch=50.left=8.right=11.newcost.json'
                        )
    parser.add_argument("--cache_file", type=str,
                        default='cache_tasks.12982.json')
    
    parser.add_argument("--left", type=int, default=0)
    parser.add_argument("--right", type=int, default=10000)

    args = parser.parse_args()
    printf(f"ENV>: {args}")

    data = load_data(args.input_file)[args.left: args.right]
    difficulty = defaultdict(int)
    for line in data:
        d = mean([task['difficulty'] for task in line['tasks']])
        difficulty[int(d)] += 1
    printf(f"ENV>: {difficulty}")

    team = [
        'Mistral-Small-Instruct-2409', 
        'Qwen2-1.5B-Instruct', 
        'Qwen2.5-3B-Instruct', 
        'phi-4', 
        'Qwen2.5-7B-Instruct',
        'Llama-3.1-8B-Instruct',
        'gemma-2-9b-it', 
        'Qwen2.5-14B-Instruct', 
        'Qwen2.5-32B-Instruct', 
        'DeepSeek-R1-Distill-Qwen-32B',
        'Qwen2-72B-Instruct', 
        'Llama-3.1-70B-Instruct', 
    ]
    agents = [Agent(DOCUMENTATIONS[model_name]) for model_name in team]

    # instantiate an env
    kwargs = {
        "core": Agent(model_config={"base_url": "xxx", "model_name": args.persona}),
        "agents": agents,
        "output_dir": args.output_dir,
        "bad_count": args.bad,
        "max_turns": 100,
        "cache_file": args.cache_file,
        "remove": args.remove,
        "HP": args.hp
    }

    if str(args.persona) == 'efficient':
        env = StepEfficientEnv
        kwargs['topk'] = args.topk
    elif args.persona == 'ideal':
        env = IdealEnv
        kwargs['topk'] = args.topk
        kwargs['bar'] = args.bar
    elif args.persona == 'random':
        env = StepRandomEnv
    elif args.persona == 'greedy':
        env = GreedyEnv
        kwargs['reward_fn'] = Reward.reward_fn
    elif args.persona == 'fair':
        env = StepRandomFairEnv
    elif args.persona == 'efair':
        env = StepEfficientFairEnv
    elif args.persona == 'optimal':
        env = PosteriorOptimalEnv
        kwargs['reward_fn'] = Reward.reward_fn
    else:
        raise ValueError(f"Invalid persona: {args.persona}")

    env = env(**kwargs)

    bar = tqdm(data)

    avg_ROI = 0.0
    avg_Gini = 0.0
    avg_Reward = 0.0
    avg_cost = 0.0
    cnt = 0
    remain = []
    survive_rate = 0.0
    step_n = 0.0

    process_gini, process_roi, process_name = [], [], []
    final_gini, final_roi = [], []
    swf = 0.0
    for i, batch in enumerate(bar, 1):
        tasks = [
            Task(type=line['type'],
                raw=line['problem'] + f" (difficulty: {line['difficulty']} / 12)", 
                content=line['problem'] + f" (difficulty: {line['difficulty']} / 12)", 
                ground_turth=line['solution'],
                task_id=line['task_id'],
                difficulty=line['difficulty'])
            for line in batch['tasks']
        ]
        printf("\nTASK>\n"+'\n'.join(task.raw for task in tasks))
        outputs = env.run(tasks, pre_reset=True, reward_fn = Reward.reward_fn)
        batch['trajectory'] = outputs

        avg_Reward += outputs['trajectory'][-1]['record']['total_reward']
        avg_cost += outputs['trajectory'][-1]['record']['total_cost']
        avg_ROI += outputs['trajectory'][-1]['record']['total_ROI']
        avg_Gini += outputs['trajectory'][-1]['record']['total_Gini']
        swf += (1- outputs['trajectory'][-1]['record']['total_Gini']) * outputs['trajectory'][-1]['record']['total_ROI']
        final_roi.append(outputs['trajectory'][-1]['record']['total_ROI'])
        final_gini.append( outputs['trajectory'][-1]['record']['total_Gini'])

        cnt += 1
        survive_rate += outputs['agents'] / len(agents)
        step_n += outputs['step_n']
        # exit()
        
        bar.set_postfix({
            "batch_id": batch['batch_id'],
            "(1-Gini) * ROI": round(sum([(1 - g) * r for g, r in zip(final_gini, final_roi)]) / len(final_gini), 3),
            "agents": round(survive_rate / cnt, 3),
            "avg_ROI": round(avg_ROI / cnt, 3),
            "avg_Gini": round(avg_Gini / cnt, 3),
            "avg_Reward": round(avg_Reward / cnt, 3),
            "accumuated ROI": round(avg_Reward / avg_cost, 3),
            "step_n": round(step_n / cnt, 3)
        })
        remain.append(batch)
        env.write_file(f"{batch['batch_id']}", batch)
    print(cnt)
    print("TASK>", round(avg_ROI / cnt, 2), round(avg_Gini / cnt, 2), round(avg_Reward / cnt, 2), avg_cost, avg_Reward, round(swf / cnt, 2))
    file = args.input_file.replace(f"{len(data)}", f"{len(remain)}")
