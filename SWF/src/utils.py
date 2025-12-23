from openai import OpenAI
import copy
from typing import Dict, List
import os
from bs4 import BeautifulSoup
import re
import requests
import json
from json_repair import repair_json
import numpy as np

def printf(*texts):
    if os.environ.get('DEBUG', 'True') == 'False':
        return
    if texts[0].startswith('TASK>'):
        format = "\033[1;34m{text}\033[0m"
    elif texts[0].startswith("AGENT>"):
        format = "\033[32m{text}\033[0m"
    elif texts[0].startswith('ROUND>') or texts[0].startswith('==='):
        format = "\033[1;33m{text}\033[0m"
    elif texts[0].startswith('ERROR>') or texts[0].startswith('ANSWER>') or texts[0].startswith('ENV>'):
        format = "\033[1;31m{text}\033[0m"
    else:
        format = "\033[1;37m{text}\033[0m"
    for text in texts:
        print(format.format(text=text))


def load_data(filename, file_type=None):
    if file_type is None:
        file_type = ['json']
    if filename.endswith('.jsonl'):
        return [json.loads(line) for line in open(filename)]
    elif filename.endswith('.json'):
        return json.load(open(filename, 'r'))
    elif os.path.isdir(filename):
        files = [os.path.join(filename, f) for f in os.listdir(filename) if any(f.endswith(ext) for ext in file_type)]
        return [load_data(f) for f in files]
    elif filename.endswith('.txt'):
        with open(filename, 'r') as f:
            data = [line.strip() for line in f]
            return data
    else:
        raise "no suitable function to load data"


def write_file(filename, data):
    if filename.endswith('.jsonl'):
        with open(filename, 'w') as f:
            for line in data:
                f.write(json.dumps(line) + '\n')
    elif filename.endswith('.txt'):
        with open(filename, 'w') as f:
            for line in data:
                f.write(line + '\n')

    elif filename.endswith('.json'):
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        raise "no suitable function to write data"

def _parse(outputs: str, tag="task"):
    soup = BeautifulSoup(outputs, 'lxml')
    agents = soup.find_all(re.compile(rf'{tag}\d*', re.IGNORECASE))
    extracted_texts = [agent.get_text(strip=True).strip() for agent in agents]
    if extracted_texts == []:
        tmp = outputs.split('\n')
        for line in tmp:
            for i in range(0, 10):
                line = line.replace(f'agent {i}', '').replace(f'agent{i}','').strip()
        return tmp
    return extracted_texts

def extract_json(content):
    try:
        tmp = json.loads(repair_json(content))
        if isinstance(tmp, dict):
            return True, tmp
        elif isinstance(tmp, list):
            tmp = [e for e in tmp if isinstance(e, dict)]
            return True, tmp[0]
        else:
            return False, content
    except:
        return False, content


def extract_boxed_answer(solution_string: str) -> list:
    """
    从包含数学解题过程的字符串中提取所有 \boxed{} 标签内的答案。

    Args:
        solution_string: 包含解题过程的字符串，可能包含 \boxed{} 标签。

    Returns:
        一个列表，包含所有找到的 \boxed{} 标签内的答案。如果未找到，则返回空列表。
    """
    pattern = r"\\boxed\{(.*?)\}"
    answers = re.findall(pattern, solution_string)
    return answers


api_search = "http://11.219.2.7:8893"  # default retriever port

def wiki_search(query, top_k=3):
    url = f'{api_search}/api/search?query={query}&k={top_k}'
    response = requests.get(url)
    res = response.json()
    knowledge = []
    for doc in res['topk']:
        text = doc['text'][doc['text'].index('|') + 1:].replace('"', '').strip()
        title = doc['text'][:doc['text'].index('|')].replace('"', '').strip()
        knowledge.append(f"Title: {title}. Content: {text}")
    return knowledge




MODELS = [
    {
        "model_name": "Mistral-Small-Instruct-2409",
        "name": "Mistral24B",
        "index": "AAA",
        # "description": "Mistral-Small-Instruct-2409 is a lightweight model that balances performance with efficiency. It costs 24 per token.",
        # "cost": 24,
        "Average": 29.92,
        "IFEval": 62.83,
        "BBH": 40.56,
        "MMLU": 20.39,
        "GPQA": 11.07,
        "MuSR": 10.23,
        "MATH": 34.43,
        'base_url': [],
        'api_key': "EMPTY",
        "speed": [(4557.78, 4)],
        "Country": "France (Europe)",
        "Origin": "Mixtral Company"
    },
    {
        "model_name": "Qwen2-1.5B-Instruct",
        "name": "Qwen1.5B",
        "index": "DDD",
        # "description": "Qwen2-1.5B-Instruct is a small model that performs relatively well across various tasks.",
        # "cost": 1.5,
        "Average": 14.14,
        "IFEval": 33.71,
        "BBH": 13.70 ,
        "MATH": 7.18,
        "GPQA": 1.57,
        "MuSR": 12.03,
        "MMLU": 16.68,
        'base_url': [],
        'api_key': "EMPTY",
        "speed": [(19952.18, 1), (17497.67 , 1)],
        "Country": "China",
        "Origin": "Alibaba cloud"
    },
    {
        "model_name": "Qwen2.5-3B-Instruct",
        "name": "Qwen3B",
        "index": "EEE",
        # "description": "Qwen2.5-3B-Instruct is a small model that performs relatively well across various tasks.",
        # "cost": 3,
        "Average": 27.16,
        "IFEval": 64.75,
        "BBH": 25.80,
        "MATH": 36.78,
        "GPQA": 3.02,
        "MuSR": 7.57,
        "MMLU": 25.05,
        'base_url': [],
        'api_key': "EMPTY",
        "speed": [(12680.62, 1), (12583.20, 1)],
        "Country": "China",
        "Origin": "Alibaba cloud"
    },
    {
        "model_name": "phi-4",
        "name": "phi4",
        "index": "FFF",
        # "description": "phi-4 is a small model that performs relatively well across various tasks.",
        # "cost": 7,
        "Average":  41.76,
        "IFEval": 69.00,
        "BBH": 55.80,
        "MATH": 46.37,
        "GPQA": 13.53,
        "MuSR": 16.68,
        "MMLU": 49.15,
        'base_url': [],
        'api_key': "EMPTY",
        "speed": [(4523.51, 2)],
        "Country": "United States American (USA)",
        "Origin": "Microsoft"
    },
    {
        "model_name": "Qwen2.5-7B-Instruct",
        "name": "Qwen7B",
        "index": "HHH",
        # "description": "Qwen2.5-7B-Instruct is a small model that performs relatively well across various tasks.",
        # "cost": 7,
        "Average": 35.20,
        "IFEval": 75.85,
        "BBH": 34.89,
        "MATH": 50.00,
        "GPQA": 5.48,
        "MuSR": 8.45,
        "MMLU": 36.52,
        'base_url': [],
        'api_key': "EMPTY",
        "speed": [(7942.57, 1)],
        "Country": "China",
        "Origin": "Alibaba cloud"
    },
    { 
        "model_name": "Llama-3.1-8B-Instruct",
        "name": "Llama8B",
        "index": "III",
        # "description": "Llama-3.1-8B-Instruct is a small model that performs relatively well across various tasks.",
        # "cost": 8,
        "Average": 23.76,
        "IFEval": 49.22,
        "BBH": 29.38,
        "MATH": 15.56,
        "GPQA": 8.72,
        "MuSR": 8.61,
        "MMLU": 31.09,
        'base_url': [],
        'api_key': "EMPTY",
        "speed": [(7005.68, 1)],
        "Country": "United States American (USA)",
        "Origin": "Meta"
    },
    {
        "model_name": "gemma-2-9b-it",
        "name": "Gemma9B",
        "index": "JJJ",
        # "description": "Gemma9B is a model that performs relatively well across various tasks.",
        # "cost": 9,
        "Average":  32.07,
        "IFEval": 74.36,
        "BBH": 42.14,
        "MATH": 19.49,
        "GPQA": 14.77,
        "MuSR": 9.74,
        "MMLU": 31.95,
        'base_url': [],
        'api_key': "EMPTY",
        "speed": [(6518.22, 2), (8170.07, 2)],
        "Country": "United States American (USA)",
        "Origin": "Google & DeepMind"
    },
    {
        "model_name": "Qwen2.5-14B-Instruct",
        "name": "Qwen14B",
        "index": "KKK",
        # "description": "Qwen2.5-14B-Instruct is a model that performs relatively well across various tasks.",
        "cost": 14,
        "Average": 41.31,
        "IFEval": 81.58,
        "BBH": 48.36,
        "MATH": 54.76,
        "GPQA": 9.62,
        "MuSR": 10.16,
        "MMLU": 43.38,
        'base_url': [],
        'api_key': "EMPTY",
        "speed": [(6518.22, 2), ],
        "Country": "China",
        "Origin": "Alibaba cloud"
    },
    {
        "model_name": "Qwen2.5-32B-Instruct",
        "name": "Qwen32B",
        "index": "LLL",
        # "description": "Qwen2.5-32B-Instruct is a model that performs relatively well across various tasks.",
        # "cost": 32,
        "Average": 46.60,
        "IFEval": 83.46,
        "BBH": 56.49,
        "MATH": 62.54,
        "GPQA": 11.74,
        "MuSR": 13.50,
        "MMLU": 51.85,
        'base_url': ['http://11.222.154.165:8081/v1', 'http://11.222.154.165:8082/v1'],
        'api_key': "EMPTY",
        "speed": [(5478.04, 4), (5901.12, 4)],
        "Country": "China",
        "Origin": "Alibaba cloud"
    },
    {
        "model_name": "DeepSeek-R1-Distill-Qwen-32B",
        "name": "DSQwen32B",
        "index": "MMM",
        # "description": "Qwen32B is a model that performs well across various tasks.",
        # "cost": 32,
        "Average": 22.96,
        "IFEval": 41.86,
        "BBH": 17.15,
        "MATH": 17.07,
        "GPQA":4.59,
        "MuSR":16.14,
        "MMLU": 40.96,
        'base_url': [],
        'api_key': "EMPTY",
        "speed": [(3947.77, 4), (5624.25, 4), (6095.65, 4)],
        "Country": "China",
        "Origin": "DeepSeek Company"
    },
    {
        "model_name": "Qwen2-72B-Instruct",
        "name": "Qwen72B",
        "index": "OOO",
        # "description": "Qwen72B-Instruct is a model that performs well across various tasks. It costs 72 per token.",
        # "cost": 72,
        "Average": 43.59 ,
        "IFEval": 79.89,
        "BBH": 57.48 ,
        "MATH": 41.77,
        "GPQA": 16.33,
        "MuSR": 17.17,
        "MMLU": 48.92,
        'base_url': [],
        'api_key': "EMPTY",
        "speed": [(4157.55, 8), (4201.77, 8)],
        "Country": "China",
        "Origin": "Alibaba cloud"
    },
    {
        "model_name": "Llama-3.1-70B-Instruct",
        "name": "Llama70B",
        "index": "PPP",
        # "description": "Llama70B is a model that performs well across various tasks. It costs 72 per token.",
        # "cost": 70,
        "Average":  43.41,
        "IFEval": 86.69,
        "BBH": 55.93,
        "MATH": 38.07,
        "GPQA": 14.21,
        "MuSR": 17.69,
        "MMLU": 47.88,
        'base_url': [],
        'api_key': "EMPTY",
        "speed": [(5279.23, 8), (5097.09, 8)],
        "Country": "United States American (USA)",
        "Origin": "Meta"
    },
]

ID = os.environ.get('ID', 'name')
if ID == 'index':
    printf("AGENT> Using the index instead of name as ID...")
    for config in MODELS:
        config['name'] = config.get('index', config['name'])
        printf(config['name'])


DOCUMENTATIONS = {
    line['model_name']: line for line in MODELS
}

DOCUMENTATIONS['baseline'] = {
    "model_name": "baselines",
    "name": "baselines",
    "Country": "None",
    "Origin": "None"
}


DOCUMENTATIONS_R = {
    line['name']: line for line in MODELS
}

LEADERBOARD = {}

LEADERBOARD['IFEval'] = """Test the model's ability to follow explicit formatting instructions.
Scoring: Accuracy – Was the format strictly followed as requested?"""

LEADERBOARD['BBH'] = """A collection of challenging tasks for members across domains, such as language understanding, mathematical reasoning, common sense, and world knowledge.
Scoring: Accuracy – Was the correct choice selected from the available options?"""

LEADERBOARD['MATH'] = """High school-level mathematical problems, including topics such as complex algebra, geometry, and advanced calculus.
Scoring: Exact match – Was the solution both correct and in the expected format?"""

LEADERBOARD['GPQA'] = """PhD-level multiple choice questions across science domains, such as chemistry, biology, and physics.
Scoring: Accuracy – Was the correct choice selected from the available options?"""

LEADERBOARD['MuSR'] = """Multistep soft reasoning (MuSR) task. Involves reasoning and understanding of long texts, such as language understanding, reasoning capabilities, and long-context reasoning.
Scoring: Accuracy – Was the correct choice selected from the available options?"""

LEADERBOARD['MMLU'] = """Massive Multitask Language Understanding – Professional (MMLU-Pro). A collection of expertly reviewed multiple choice questions across domains such as medicine, healthcare, law, ethics, engineering, and mathematics.
Scoring: Accuracy – Was the correct choice selected from the available options?"""

LEADERBOARD['CO2'] = """Overview: Evaluation of CO2 emissions associated with members inference. This score focuses on model inference within our specific setup, considering factors like data center location and energy mix. It allows for the equivalent comparison of models in our use case.
Why it matters: Large models can have significant carbon footprints, and this evaluation helps make informed choices about model selection based on their environmental impact."""

LEADERBOARD['Average'] = """Purpose: General capability evaluation score."""

LEADERBOARD['Country'] = """Purpose: Country of the members."""

LEADERBOARD['Origin'] = """Purpose: Origin of the members."""



def load_leaderboard(
        names: List[str],
        leaderboards: List[str] = ["IFEval", "MATH", "GPQA", "MuSR", "MMLU", "Average"],
        width: int = 10,
    ):
    header = ["name"] + leaderboards
    collects = [header]
    for name in names:
        collect = [DOCUMENTATIONS_R[name][col] for col in header]
        collects.append(collect)
    
    collects = [
        ' '.join([collect[0].rjust(width+2)] + [str(e).ljust(width) for e in collect[1:]])
        for collect in collects
    ]
    collects = '\n'.join(collects)
    collects += "\n\nCaption: **Overview** of each column." + '\n\n' + '\n\n'.join([f'**{col}**: ' + LEADERBOARD[col] for i, col in enumerate(leaderboards, 1)])
    collects += f"\nYou can only select from {names} with HP > 0 (enclosed in <agent> <agent_name>)."
    return collects


def gini_coefficient(wealth):
    # 确保财富数组是从小到大排序的
    wealth = np.sort(wealth)
    
    # 计算财富的总和
    total_wealth = np.sum(wealth)
    
    # 计算基尼系数
    n = len(wealth)
    cumulative_wealth = np.cumsum(wealth)
    
    if total_wealth == 0:
        return 0
    # 使用公式计算基尼系数
    gini = (n + 1 - 2 * np.sum(cumulative_wealth) / total_wealth) / n
    
    return gini

def theil_index(incomes):
    """
    计算给定收入数据的泰尔指数。
    
    参数:
    incomes (list or np.array): 收入数据
    
    返回:
    float: 泰尔指数
    """
    incomes = np.array(incomes)
    total_income = np.sum(incomes)
    mean_income = total_income / len(incomes)
    
    T = np.sum((incomes / total_income) * np.log(total_income / incomes))
    
    return T

def coefficient_of_variation(data):
    """
    计算给定数据的变异系数（Coefficient of Variation）。
    
    参数:
    data (list or np.array): 数据
    
    返回:
    float: 变异系数（以百分比表示）
    """
    data = np.array(data)
    mean = np.mean(data)  # 计算均值
    std_dev = np.std(data)  # 计算标准差
    
    # 计算变异系数
    cv = (std_dev / mean) * 100  # 以百分比表示
    return cv


index2model = {model['index']: model['model_name']  for model in MODELS if 'index' in model}
name2model = {model['name']: model['model_name']  for model in MODELS if 'name' in model}