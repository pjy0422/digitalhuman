import os
from tqdm import tqdm
from typing import List, Dict
import sys
sys.path.append('../src')
from base import Agent, Task, Response, CacheEnv
from utils import _parse, load_data, printf, load_leaderboard, DOCUMENTATIONS
from prompt import *
from metric import *
import argparse
from dataclasses import dataclass, field
import multiprocessing
import random
import copy
from tabulate import tabulate

@dataclass
class StreamOrdinator(Agent):
    leaderboards: List[str] = field(default_factory=lambda: ["IFEval", "MATH", "GPQA", "MuSR", "MMLU", "Average"])

    def generate(self, messages, **kwargs):
        if self.model_name == 'DeepSeek-R1':
            kwargs['temperature'] = 0.5
        return super().generate(messages, **kwargs)

    def step(self, tasks: List[Task], team: List[str],**kwargs):
        """
        Args:
            tasks: stringfied list of tasks
            active_task_ids: list of active task ids
            active_agent_ids: list of active agent ids
        """
        leaderboard_str = load_leaderboard(names=team, leaderboards=self.leaderboards)
        # _extra = ['Country'] if self.culture else []
        # if self.culture:
        #     leaderboard_str += "\nPlease note that some team member from the same country with you while some are not."
        system_prompt = self.system.format(team=leaderboard_str, num=len(team))
        # if self.culture:
        #     system_prompt = f"You are {self.name} from {self.origin}, {self.country}." + system_prompt

        messages = [
            {"role": "system", "content": system_prompt},
        ]
        if len(self.history) == 0:
            self.history.append({"role": "user", "content": self.inputs.format(task=tasks[0].content)})
        else:
            assert self.history[-1]['role'] == 'user' and len(messages) % 2 == 1
            self.history[-1]['content'] += '\n\n' + self.inputs.format(task=tasks[0].content)

        messages += self.history[-7:]
        if 'gpt-5' in self.model_name and 'chat' not in self.model_name or os.environ.get('SHORT', 'False') == 'True':
            messages[-1]['content'] += '\n Based on the above requirement, please select one member and enclose the corresponding name in <agent> </agent>. Please **very briely** explain your reasoning (less than 500-600 words).**'
        elif os.environ.get('SHORT','False') == 'Extreme':
            print('=' * 100)
            messages[-1]['content'] += '\n Based on the above requirement, please select one member and enclose the corresponding name in <agent> </agent>. You can only summarize your reasoning in **only one concise sentence**.'

        print(messages[0]['content'])
        print(messages[-1]['content'])
        cnt = 0
        while cnt < self.max_retry:
            outputs: Dict = self.generate(messages)
            try:
                names = _parse(outputs['content'], tag='agent')
                names = [name for name in names if name in team]
                self.usage.append(outputs['usage'])
                return Response(end=True,
                                raw=outputs['content'], 
                                content=names[-1], 
                                role='ordinator',
                                usage=outputs['usage'],
                                cost=outputs['cost'])
            except:
                cnt += 1
                messages.append({"role": "assistant", "content": outputs['content']})
                messages.append({"role": "user", "content": f"You select no valid member. Please reading the context carefully and select one from '{team}' enclosed in <agent> </agent>."})

        printf('TASK> Using random selection')
        return Response(end=False,
                        raw="Error in parsing, using default random selection", 
                        content=random.choice(team), 
                        role='ordinator',
                        usage=outputs['usage'],
                        cost=outputs['cost']*3)

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

def _run(args, data):

    if args.persona == 'must fair with penalty':
        system_prompt = CORE_SYSTEM_GENERAL_PROMPT_MUST_FAIR_PENTALTY
        input_prompt = CORE_INSTRUCTIONS_GENERAL_PROCESS
    elif args.persona == 'must fair with benefit':
        system_prompt = CORE_SYSTEM_GENERAL_PROMPT_MUST_FAIR_BENEFIT
        input_prompt = CORE_INSTRUCTIONS_GENERAL_PROCESS
    elif args.persona == 'general identification':
        system_prompt = CORE_SYSTEM_GENERAL_PROMPT_IDENTIFICATION
        input_prompt = CORE_INSTRUCTIONS_GENERAL_PROCESS
    elif args.persona == 'general internalization':
        system_prompt = CORE_SYSTEM_GENERAL_PROMPT_INTERNALIZATION
        input_prompt = CORE_INSTRUCTIONS_GENERAL_PROCESS
    else:
        if args.remove:
            system_prompt = CORE_SYSTEM_GENERAL_PROMPT_W_REMOVE
        else:
            system_prompt = CORE_SYSTEM_GENERAL_PROMPT
        input_prompt = CORE_INSTRUCTIONS_GENERAL_PROCESS

    recipients = [Agent(DOCUMENTATIONS[model]) for model in team]
    model_config = {
        "model_name": args.model_name,
        "base_url": args.base_url,
        "api_key": args.api_key,
    }

    allocator = StreamOrdinator(
        system=system_prompt,
        inputs=input_prompt,
        model_config=model_config,
        max_timesteps=1000,
        max_retry=10,
    )

    env = CacheEnv(
        core=allocator,
        agents=recipients,
        output_dir=args.output_dir,
        max_turns=args.max_turns,
        budget=100,
        prefix='TASK',
        remove=args.remove,
        HP=args.hp if args.hp != -1 else len(recipients) - 1,
        cache_file='../env/cache_tasks.12982.json',
    )

    bar = tqdm(data)

    for _, batch in enumerate(bar):
        batch['config'] = vars(args)
        tasks = [
            Task(type=line['type'],
                raw=line['problem'] + f" (difficulty: {line['difficulty']} / {len(team)})", 
                content=line['problem'] + f" (difficulty: {line['difficulty']} / {len(team)})", 
                ground_turth=line['solution'],
                task_id=line['task_id'],
                difficulty=line['difficulty'])
            for line in batch['tasks']
        ]
        trajectory = env.run(
            tasks,
            pre_reset=True,
            reward_fn=Reward.reward_fn,
        )
        batch['trajectory'] = trajectory
        env.write_file(f"{batch['batch_id']}", batch)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--max_turns", type=int, default=100)
    parser.add_argument("--persona", type=str, default='general')
    parser.add_argument("--hp", type=int, default=5)
    parser.add_argument("--remove", action="store_true", default=False, help="Enable removal option")
    parser.add_argument("--input_file", type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = load_data(args.input_file)
    cache = load_data(args.output_dir)
    cache = [line['batch_id'] for line in cache]
    data = [line for line in data if line['batch_id'] not in cache]

    print(f'reload cache with {len(cache)} items. Running the remaining {len(data)} items.')

    _run(args, data)
    exit(0)
    num_processes = min(5, len(data))

    if len(data) < 5:
        _run(args, data)
        exit()

    length = len(data) // num_processes + 1
    pool = multiprocessing.Pool(processes=num_processes)

    collects = []
    for ids in range(num_processes):
        collect = data[ids * length:(ids + 1) * length]
        collects.append(pool.apply_async(_run, (args, collect)))
    pool.close()
    pool.join()

    print('All done.')

if __name__ == "__main__":
    main()
