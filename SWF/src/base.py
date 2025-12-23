import json
import re, string, os, sys
from collections import defaultdict
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from typing import List, Dict, Any, Union, Callable
import os
import copy
from openai import OpenAI
from utils import printf
from dataclasses import dataclass, field
import random
from tabulate import tabulate
import numpy as np
import time
from utils import gini_coefficient

@dataclass
class Task:
    type: str # the type of the task
    raw: str # the raw generated content of the task
    content: str # the extracted content of the raw content
    ground_turth: str # the ground truth of the task
    task_id: str = field(default="DEFAULT_ID")  # default value for task_id
    difficulty: int = field(default=2)  # default value for task_id

@dataclass
class Response:
    role: str # ordinator or sub agent
    raw: str # the raw content of the response
    content: Union[str, Dict] # the extracted content of the raw content
    end: bool # True for ending, False for not ending
    usage: List[int]
    cost: float


@dataclass
class Agent:
    model_config: Dict
    max_timesteps: int = 10
    max_retry: int = 5
    system: str = ""
    inputs: str = ""
    # culture: bool = False

    def __post_init__(self) -> None:
        self.base_urls = self.model_config['base_url']
        self.api_key = self.model_config.get('api_key', 'EMPTY')
        self.time_stamp = 0
        self.history = []
        self.usage = []
        self.memory = defaultdict(list)

    @property
    def model_name(self):
        return self.model_config['model_name']

    @property
    def name(self):
        return self.model_config['name']

    def is_halted(self) -> bool:
        return self.time_stamp > self.max_timesteps

    def is_finished(self, content) -> bool:
        # print(content)
        if '<answer>' in content and '</answer>' in content:
            return True
        return False

    # def step(self, tasks: List[Task], *args, **kwargs):
    #     raise NotImplementedError

    @property
    def cost(self) -> float:
        """
        cost = number of gpu / (k token / s) = number of gpu * s / k token
        How much compute (GPU seconds) is needed to run 1k tokens
        """
        speed = self.model_config.get('speed', [])
        if len(speed) == 0:
            raise ValueError("speed record is empty")
        avg_speed = sum([x[1] / (x[0] / 1000) for x in speed]) / len(speed)
        return avg_speed
    

    def run(self, inst: str, pre_reset=True) -> Response:
        """
        inst: instruction
        """
        raise NotImplementedError

    def generate(self, messages, **kwargs) -> Dict: #TODO
        """
        call LLM to generate response based on the history messages
        
        Args:
            messages: the history messages
        
        Returns:
            the generated response
        
        Please customize the LLM call in this function, e.g., Anthropic API, OpenAI API, Gemini API, HunYuan API, etc.
        By default, we use *OpenAI API* to generate response.
        """
        _kwargs = copy.deepcopy(kwargs)
        _kwargs['model'] = self.model_name
        _kwargs['messages'] = messages
        llm = OpenAI(
            api_key=self.api_key,
            base_url=self.base_urls,
        )
        response = llm.chat.completions.create(
            **_kwargs
        )
        content = response.choices[0].message.content
        usage = [response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens]
        return {"content": content, "usage": usage, "cost": 0}

    def update(self, messages, key=None):
        if key is None:
            key = len(self.memory)
        self.memory[key].extend(messages)

    def append(self, role, content):
        self.history.append({"role": role, "content": content})

    def reset(self):
        self.time_stamp = 0
        self.history = []
        self.memory = defaultdict(list)
        self.usage = []


@dataclass
class BaseEnv:
    core: Agent
    agents: List[Agent] # name, not the real model name
    output_dir: str
    max_turns: int
    max_retry: int = 5
    budget: float = 100.0
    prefix: str = 'query'
    HP: float = 3.0
    remove: bool = False
    header: List[str] = field(default_factory=lambda: ['HP'])

    def __post_init__(self):
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        except:
            raise f"{self.output_dir} does not exist"
        self.name2idx = {agent.name: idx for idx, agent in enumerate(self.agents)}
        self.map = {agent.name: agent.model_name for agent in self.agents}

        self.records = {
            agent.name: defaultdict(list)
            for agent in self.agents
        }
        for name, record in self.records.items():
            record['HP'] = [self.HP]
        self.step_n = 0
        self.reset()


    def reset(self):
        self.core.reset()
        for agent in self.agents:
            agent.reset()
        self.step_n = 0
        self.records = {
            agent.name: defaultdict(list)
            for agent in self.agents
        }
        for name, record in self.records.items():
            record['HP'] = [self.HP]

    def is_halted(self) -> bool:
        return (self.step_n > self.max_turns)

    def is_finished(self, outputs) -> bool:
        if "<answer>" in outputs and "</answer>" in outputs:
            return True
        return False

    def _task_assign(
        self,
        tasks: List[Task],
        survived_agent_names: set,
        **kwargs
    ):
        # remove the eliminated
        # _team = [name for name, record in self.records.items() if name in survived_agent_names]
        _team = list(survived_agent_names)
        response: Response = self.core.step(tasks, team=_team, **kwargs)
        return response

    def get_cost(self, usages):
        return (usages[0]*1 + usages[1]*4) / 10000

    def update(self, name, **kwargs):
        for k, v in kwargs.items():
            self.records[name][k].append(float(v))
            # For other agents, if they has not been used, here we add the same key to their record with an empty list.
            for agent in self.agents:
                _name = agent.name
                if _name != name and k not in self.records[_name]:
                    self.records[_name][k] = []

    def get_file(self, idx):
        return f'{self.prefix}_{idx}.json'

    def write_file(self,
        idx,
        messages
    ):
        json.dump(messages, open(os.path.join(self.output_dir, self.get_file(idx)), 'w'), indent=4)

    def inference_from_resume(self, data, key) -> List:
        cache =[json.load(os.path.join(self.output_dir, file)) for file in os.listdir(self.output_dir) if file.endswith('.json')]
        cache = [line[key] for line in cache]
        _data = [line for line in data if line[key] not in cache]
        return _data
    
    def get_records(self, survived_agent_names: set):
        def _sum(arr):
            # if len(arr) == 0:
            #     return 'None'
            return round(sum(arr), 3)
        
        def _mean(arr):
            if len(arr) == 0:
                return "None"
            return round(sum(arr)/len(arr), 3)
        
        def _review(value, avg):
            LOW = "Under team average"
            UP = "Beyond team average"
            return 'None' if not isinstance(value, float) and not isinstance(value, int) else LOW if value <= avg else UP 
        
        team = [name for name in self.records.keys() if name in survived_agent_names]
        
        team_reward = [_sum(self.records[name]['reward']) for name in team]
        team_cost = [_sum(self.records[name]['cost']) for name in team]
        num_tasks = [len(self.records[name]['cost']) for name in team]

        total_reward = sum(team_reward)
        total_cost = sum(team_cost)
        avg_roi = total_reward / total_cost if total_cost !=0 else 0
        gini = gini_coefficient(num_tasks)

        overall_records = """\nDue to the limited space, the summarized overall review of history task-solving is as follows:
The total cost is {cost:.2f}.
The totol reward is {rewards:.2f}. 
The overal ROI (e.g., reward / cost) is {ROI:.2f}.
The Gini Coefficient is {gini:.2f}.""".format(cost=total_cost, rewards=total_reward, ROI=avg_roi, gini=gini)
        
        ROI = [_sum(self.records[name]['reward']) / _sum(self.records[name]['cost']) if len(self.records[name]['cost']) else 'None' 
                    for name in team]
        review = [_review(e, avg_roi) for e in ROI]
        table = {
            "Name": team,
            "HP": [_sum(self.records[name]['HP']) for name in team],
            "Number of tasks": num_tasks,
            "Total Reward": team_reward,
            "Cost": team_cost,
            "Avg. Delivery Rate": [_mean(self.records[name]['delivery_rate']) for name in team],
            "Avg. Successful Rate":  [_mean(self.records[name]['reward']) for name in team],
            "ROI": ROI,
            "ROI Review": review
        }
    
        caption = [
            '**Name**: The name of the team member with HP >= 0. You can only assign the task to them since the members with HP < 0 has been eliminated.'
            "**HP**: The health points of the team member, representing their overall health status.",
            "**Number of tasks**: The total number of tasks assigned to the team member.",
            "**Total Reward**: The total reward that the team member has earned from completing assigned tasks.",
            "**Avg. Cost**: The average cost per task assigned to the team member, calculated as the total cost divided by the number of tasks.",
            "**Avg. Delivery Rate**: The average delivery rate of tasks. We set the delivery rate of task to '1' if task was answered (correct or incorrect); '0' means no answer was provided.",
            "**Avg. Reward**: The average reward the team member has received per task, calculated as the total reward divided by the number of tasks.",
            "**ROI**: The productivity of the team member, calculated as the ratio of total reward to total cost, reflecting the efficiency of task completion."
        ]

        # if self.core.culture:
            # table['Country'] = [self.agents[self.name2idx[name]].country for name in team]
            # caption.append("**Country & Origin**: The home country of the team member and the institution that recommended them. Some of them come from the same country or institution with your own; while others not")

        table_str = tabulate(table, headers="keys",tablefmt='pipe', showindex=True)
        
        caption_str = 'Caption: The history track record used to evaluate your team member. "None" is the default value, which indicates that this model has not been used before.\n'
        caption_str += '\n'.join(caption)

        table['total_Gini'] = gini
        table['total_reward'] = total_reward
        table['total_cost'] = total_cost
        table['total_ROI'] = avg_roi

        return overall_records + '\nStatistics of each **remaining** team member (HP > 0) are evaluated as below:\n' + table_str + '\n\n' + caption_str, table

    def run(self,
        tasks: List[Task], 
        pre_reset=True,
        reward_fn=None,
    ):
        raise NotImplementedError

@dataclass
class CacheEnv(BaseEnv):
    cache_file: str = ''
    bad_count: int = 6 # type: ignore

    def __post_init__(self):
        super().__post_init__()
        self.cache = self.load_cache()

    def get_cost(self, usages):
        """
        1/100: scaling factor
        * / 1000: total k token 
        """
        return 0.5 * ((usages[0]*1 + usages[1]*1) / 1000)
    
    def load_cache(self) -> Dict[str, Dict[str, Any]]:
        """
        Read the task output of a single agent
        {"task_id": str, "task": str, "output": "long output", "answer": "short answer", "usage": List[int], }
        """
        
        return json.load(open(self.cache_file, 'r'))

    def look_up_cache(self, task_id, agent_name) -> Response:
        """
        Look up task output in cache
        """
        model_name = self.map[agent_name]
        responses = self.cache.get(task_id, {}).get(model_name, None)
        idx = random.randint(0, 100000) % len(responses)
        response = responses[idx]
        if response is None:
            return None
        agent_resp = Response(
            role=response['role'],
            raw=response['raw'],
            content=response['content'],
            end=response['end'],
            usage=response['usage'],
            cost=0
        )
        return agent_resp

    def get_env_feedback(self, reward_fn, response, name, task) -> dict:
        """
        Get environment feedback
        """
        c = self.get_cost(response.usage) * self.agents[self.name2idx[name]].cost
        r = reward_fn(task.type, response.content, task.ground_turth) if reward_fn else response.end
        return {'reward': r, 'cost': c}
           

    def run(self,
        tasks: List[Task],
        pre_reset=True,
        reward_fn=None,
    ):
        if pre_reset:
            self.reset()

        trajectory = []
        _tasks = copy.deepcopy(tasks)
        survived_agent_names = set([agent.name for agent in self.agents])

        bad = defaultdict(int)
        num_of_task = len(tasks)

        tmp_cost = []
        while not self.is_halted() and len(_tasks) > 0:
            # constant variable
            tmp_task = _tasks[0]
            self.step_n += 1

            # task assignment
            statistics = []
            _, _table = self.get_records(survived_agent_names)
            _table = {_n: _task_num for _n, _task_num in zip(_table['Name'], _table['Number of tasks'])}
            for _name in survived_agent_names:
                _feedback = self.get_env_feedback(reward_fn, self.look_up_cache(_tasks[0].task_id, _name), _name, _tasks[0])
                _feedback["name"] = _name
                _feedback['task'] = _table[_name]
                statistics.append(_feedback)
            response: Response = self._task_assign(_tasks, survived_agent_names, statistics=statistics)
            tmp_cost.append(response.cost)

            # task execution
            assert response.content in survived_agent_names
            agent_resp: Response = self.look_up_cache(_tasks[0].task_id, response.content)
            
            # updating the environment state
            feedback = self.get_env_feedback(reward_fn, agent_resp, response.content, _tasks[0])
            self.update(response.content, cost=feedback['cost'], delivery_rate=agent_resp.end, reward=feedback['reward'])
            
            # dealing with the removed agents
            if self.remove:
                for _name, _record in self.records.items():
                    if _name == response.content:
                        delta =  max(3*len(survived_agent_names) - sum(_record['HP']), 0) # Increase to n times the original, no need to consider for the next n rounds
                        _record['HP'].append(delta) 
                        # record['HP'].append(len(survived_agent_names)-1)
                for _name, _record in self.records.items():
                    if _name in survived_agent_names and _name != response.content:
                        _record['HP'].append(-1)
                        if sum(_record['HP']) < 0:
                            survived_agent_names.remove(_name)
                    else:
                        _record['HP'].append(0)

            # stringlize the updated environment for the next action predictoion
            env_review = f"""Feedback from Enviornment:\n\nThe finally submiited answer from {response.content} is "{agent_resp.content}", with the cost of {feedback['cost']} and the task reward of {feedback['reward']}."""
        
            if feedback['reward']!=0:
                env_review += f'\n\nGreat! The task {_tasks[0].task_id}) has been completed. Despite any imperfections, please move forward and focus on the next tasks.'
                _tasks = _tasks[1:]
            else:
                env_review = f'\n\nUnfortunately, {response.content} has not completed the task (task id of {_tasks[0].task_id}). You can reassign this task to another member within <agent></agent> tags.'
                bad[_tasks[0].task_id] += 1
                if bad[_tasks[0].task_id] >= self.bad_count: # TODO
                    _tasks = _tasks[1:]

            env_record, env_table = self.get_records(survived_agent_names)

            if self.remove and self.step_n > 3:
                env_review += f"This is the {self.step_n}th day. The previous cooperation history of you and your team is too long and has been summarized as below.\n"

            env_review += env_record

            # append the updated environment to the decision maker's context
            self.core.append(role='assistant', content=response.raw)
            self.core.append(role='user', content=env_review)

            trajectory.append({
                "name": response.content, 
                "task": tmp_task.content,
                "response": agent_resp.raw,
                "end": agent_resp.end,
                "record": env_table,
                "cost": feedback['cost'],
                "reward": feedback['reward'],
                "agents": len(survived_agent_names),
                "step_n": self.step_n
            })

            # showcase the task-solving screenshot
            # os.system('clear')
            printf(f"TASK> I am {self.core.model_name}. {response.raw}")
            # printf(f'AGENT> {response.content}\nReceive: {tmp_task.content}\nResponse: {agent_resp.content}\nGround truth: {tmp_task.ground_turth}\nReward / Cost: {feedback['reward']} / {feedback['cost']}')
            printf(f"ENV> Iteration: {self.step_n}; Process: {len(_tasks)} / {num_of_task})\n\n{env_review}")

            if len(survived_agent_names) == 1:
                break

        if len(survived_agent_names) == 1:
            mvp = [agent for agent in self.agents if agent.name in survived_agent_names][0].name
            # printf("AGENT> Only {mvp} survived.")
            while not self.is_halted() and len(_tasks) > 0:
                self.step_n += 1
                _task = _tasks[0]
                agent_resp: Response = self.look_up_cache(_task.task_id, mvp)
                c = self.get_cost(agent_resp.usage) * self.agents[self.name2idx[mvp]].cost
                r = reward_fn(_task.type, agent_resp.content, _task.ground_turth) if reward_fn else agent_resp.end
                self.update(mvp, cost=c, delivery_rate=agent_resp.end, reward=r)
                trajectory.append({
                    "name": mvp, 
                    "task": _task.content,
                    "response": agent_resp.raw,
                    "end": agent_resp.end, 
                    "record": self.get_records(survived_agent_names)[-1],
                    "cost": c,
                    "reward": r,
                    "agents": len(survived_agent_names),
                    "step_n": self.step_n
                })
                _tasks = _tasks[1:]

        return {"trajectory": trajectory, 
                "core": self.core.history, 
                "usage": self.core.usage, 
                "agents": len(survived_agent_names), 
                "step_n": self.step_n,
                "cost": tmp_cost
            }


@dataclass
class StepRandomEnv(CacheEnv):

    def _task_assign(self, tasks, survived_agent_names, statistics):
        env_record, env_table = self.get_records(survived_agent_names)
        team = env_table['Name']
        idx = random.randint(0, 100000) % len(team)
        idx = self.name2idx[team[idx]]
        if int(os.environ.get('SLEEP', 0)) > 0:
            time.sleep(int(os.environ['SLEEP']))
        
        return Response(
            role='assistant',
            raw=self.agents[idx].name,
            content=self.agents[idx].name,
            end=True,
            usage=0,
            cost=0,
        )


@dataclass
class StepEfficientEnv(CacheEnv):
    topk: int = 3
    def _task_assign(self, tasks, survived_agent_names, statistics):
        env_record, env_table = self.get_records(survived_agent_names)
        num_of_tasks = env_table['Number of tasks']
        team = env_table['Name']
        printf(f"ENV: {survived_agent_names}")
        non_used_agents = np.where(np.array(num_of_tasks) == 0)[0].tolist()
        if len(non_used_agents)!=0:
            idx = non_used_agents[random.randint(0,len(non_used_agents)-1)]
        else:
            indices = np.argsort(env_table['ROI'])[-self.topk:][::-1]
            idx = indices[random.randint(0, len(indices)-1)]

        idx = self.name2idx[team[idx]]
        if int(os.environ.get('SLEEP', 0)) > 0:
            time.sleep(int(os.environ['SLEEP']))
        return Response(
            role='assistant',
            raw=self.agents[idx].name,
            content=self.agents[idx].name,
            end=True,
            usage=0,
            cost=0,
        )

@dataclass
class GreedyEnv(CacheEnv):

    reward_fn: Callable[[], float] = lambda x, y: x==y

    def _task_assign(self, tasks, survived_agent_names,statistics):
        max_value = -1
        idx = None

        statistics = {line['name']: line for line in statistics}
        for agent in self.agents:
            _name = agent.name
            if _name not in survived_agent_names:
                continue
            agent_resp: Response = self.look_up_cache(tasks[0].task_id, _name)
            _idx = self.name2idx[_name]
            c = self.get_cost(agent_resp.usage) * self.agents[_idx].cost
            r = self.reward_fn(tasks[0].type, agent_resp.content, tasks[0].ground_turth)
            ROI = r / c
            _statistic = copy.deepcopy(statistics)
            _statistic[_name]['task'] += 1
            f = (1 - gini_coefficient([_statistic[_name]['task'] for _name in survived_agent_names])) * ROI
            if f > max_value:
                max_value = f
                idx = _idx
        if int(os.environ.get('SLEEP', 0)) > 0:
            time.sleep(int(os.environ['SLEEP']))
        # time.sleep(1)
        return Response(
            role='assistant',
            raw=self.agents[idx].name,
            content=self.agents[idx].name,
            end=True,
            usage=0,
            cost=0,
        )


@dataclass
class PosteriorOptimalEnv(CacheEnv):
    reward_fn: Callable[[], float] = lambda x, y: x==y
    topk: int = 3

    def _task_assign(self, tasks, survived_agent_names,statistics):
        max_ROI = -1
        idx = None
        for agent in self.agents:
            _name = agent.name
            if _name not in survived_agent_names:
                continue
            agent_resp: Response = self.look_up_cache(tasks[0].task_id, _name)
            _idx = self.name2idx[_name]
            c = self.get_cost(agent_resp.usage) * self.agents[_idx].cost
            r = self.reward_fn(tasks[0].type, agent_resp.content, tasks[0].ground_turth)
            ROI = r / c
            if ROI > max_ROI:
                max_ROI = ROI
                idx = _idx
        if int(os.environ.get('SLEEP', 0)) > 0:
            time.sleep(int(os.environ['SLEEP']))
        # time.sleep(1)
        return Response(
            role='assistant',
            raw=self.agents[idx].name,
            content=self.agents[idx].name,
            end=True,
            usage=0,
            cost=0,
        )


@dataclass
class StepEfficientFairEnv(CacheEnv):

    def _task_assign(self, tasks, survived_agent_names,statistics):
        env_record, env_table = self.get_records(survived_agent_names)
        num_of_tasks = env_table['Number of tasks']
        printf(f"ENV: {survived_agent_names}")
        team = env_table['Name']
        # find minimial-used agents
        minmial_used_agents = np.where(np.array(num_of_tasks) == np.min(num_of_tasks))[0].tolist()
        ROI = env_table['ROI']
        if 'None' in ROI:
            idx = minmial_used_agents[random.randint(0,100000) % len(minmial_used_agents)]
        else:
            max_ROI_idx = np.argmax(np.array(ROI)[minmial_used_agents])  # Find the index with the maximum ROI
            idx = minmial_used_agents[max_ROI_idx]  # Get the corresponding agent based on the index
        idx = self.name2idx[team[idx]]
        if int(os.environ.get('SLEEP', 0)) > 0:
            time.sleep(int(os.environ['SLEEP']))
        return Response(
            role='assistant',
            raw=self.agents[idx].name,
            content=self.agents[idx].name,
            end=True,
            usage=0,
            cost=0,
        )

@dataclass
class StepRandomFairEnv(CacheEnv):

    def _task_assign(self, tasks, survived_agent_names,statistics):
        env_record, env_table = self.get_records(survived_agent_names)
        num_of_tasks = env_table['Number of tasks']
        team = env_table['Name']

        # find minimial-used agents
        non_used_agents = np.where(np.array(num_of_tasks) == np.min(num_of_tasks))[0].tolist()
        idx = non_used_agents[random.randint(0,100000) % len(non_used_agents)]
        # time.sleep(5)
        if int(os.environ.get('SLEEP', 0)) > 0:
            time.sleep(int(os.environ['SLEEP']))
        idx = self.name2idx[team[idx]]
        return Response(
            role='assistant',
            raw=self.agents[idx].name,
            content=self.agents[idx].name,
            end=True,
            usage=0,
            cost=0,
        )


@dataclass
class IdealEnv(CacheEnv):
    """
    Default to finding the most efficient; if the Gini coefficient is too high, choose the one with the least assignment
    """
    topk: int = 3
    bar: float = 0.4

    def _task_assign(self, tasks, survived_agent_names, statistics):
        env_record, env_table = self.get_records(survived_agent_names)
        num_of_tasks = env_table['Number of tasks']
        team = env_table['Name']
        
        # find non-used agents
        non_used_agents = np.where(np.array(num_of_tasks) == 0)[0].tolist()
        # printf("AGENT>", non_used_agents)
        if len(non_used_agents)!=0:
            # idx = non_used_agents[random.randint(0,100000) % len(non_used_agents)]
            idx = non_used_agents[random.randint(0,len(non_used_agents)-1)]
        else:
            gini = gini_coefficient(num_of_tasks)
            # print(gini)
            if gini > self.bar:
                idx = np.argmin(num_of_tasks)
            else:
                indices = np.argsort(env_table['ROI'])[-self.topk:][::-1]
                idx = indices[random.randint(0, len(indices)-1)]
        if int(os.environ.get('SLEEP', 0)) > 0:
            time.sleep(int(os.environ['SLEEP']))
        idx = self.name2idx[team[idx]]
        return Response(
            role='assistant',
            raw=self.agents[idx].name,
            content=self.agents[idx].name,
            end=True,
            usage=0,
            cost=0,
        )
