import logging
import os
import json
import shutil
from tqdm import tqdm
# from agentverse.agentverse import AgentVerse
from agentverse.tasksolving import TaskSolving
from agentverse.logging import get_logger
from argparse import ArgumentParser
import asyncio
from dataloader import dataloader_registry

parser = ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    default="tasksolving/brainstorming",
)
parser.add_argument("--debug", action="store_true")
parser.add_argument(
    "--tasks_dir",
    type=str,
    # default=os.path.join(os.path.dirname(__file__), "..", "agentverse", "tasks"),
    default=''
)
parser.add_argument("--postfix", type=str, default='')
parser.add_argument(
    "--output_path",
    type=str,
    default='results/tasksolving/brainstorming'
)
parser.add_argument(
    "--skip_saving",
    action="store_true",
)
parser.add_argument("--dataset_path", type=str, default='')
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--skip_cnt", type=int, default=0)
args = parser.parse_args()


logger = get_logger()
logger.set_level(logging.DEBUG if args.debug else logging.INFO)


def get_dataloader(task, dataset_path):
    return dataloader_registry.build(task, path=dataset_path)


def cli_main():
    dataloader = get_dataloader(args.task, args.dataset_path)
    os.makedirs(args.output_path, exist_ok=True)
    shutil.copyfile(
        f"{args.tasks_dir}/{args.task}/config_{args.postfix}.yaml",
        f"{args.output_path}/config_{args.postfix}.yaml",
    )
    print(len(dataloader.examples))
    if args.debug and len(dataloader.examples) > 2:
        dataloader.examples = dataloader.examples[:2]
    print('--->', len(dataloader.examples))
    if args.skip_cnt == 0:
        skip_cnt = 0
        if not args.overwrite and os.path.exists(f"{args.output_path}/results_{args.postfix}.jsonl"):
            with open(f"{args.output_path}/results_{args.postfix}.jsonl", "r") as f:
                for line in f:
                    if line.strip():
                        skip_cnt += 1
    else:
        skip_cnt = args.skip_cnt
    f = open(f"{args.output_path}/results_{args.postfix}.jsonl", "w" if args.overwrite else "a")
    logger.info(f"Skip {skip_cnt} examples, start from {skip_cnt + 1}.")
    
    for i, example in enumerate(tqdm(dataloader, total=len(dataloader.examples))):
        if i < skip_cnt:
            continue
        logger.info(f"Input: {example['input']}\nAnswer: {example['answer']}")
        agentverse = TaskSolving.from_task(args.task, args.tasks_dir, args.postfix)
        agentverse.environment.set_task_description(example["input"])
        plan, result, logs, decision_making_process = agentverse.run()
        total_spent = agentverse.environment.get_spend()
        f.write(
            json.dumps(
                {
                    "input": example["input"],
                    "response": plan,
                    "label": example["answer"],
                    "logs": logs,
                    "decision_making_process": decision_making_process,
                    "spent": total_spent
                },  ensure_ascii=False
            )
            + "\n"
        )
        f.flush()
    f.close()


if __name__ == "__main__":
    cli_main()
