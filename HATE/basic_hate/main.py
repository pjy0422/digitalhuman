import os, json
import logging

# from agentverse.agentverse import AgentVerse
from agentverse.tasksolving import TaskSolving

# from agentverse.gui import GUI
from agentverse.logging import logger
from argparse import ArgumentParser

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
    "--overwrite",
    action="store_true",
)
parser.add_argument(
    "--skip_saving",
    action="store_true",
)
args = parser.parse_args()

logger.set_level(logging.DEBUG if args.debug else logging.INFO)


def cli_main():
    print(args.tasks_dir)
    if not args.skip_saving:
        f = open(f"{args.output_path}/results_{args.postfix}.jsonl", "w" if args.overwrite else "a")
    agentversepipeline = TaskSolving.from_task(args.task, args.tasks_dir, args.postfix)
    plan, result, logs, decision_making_process = agentversepipeline.run()
    total_spent = agentversepipeline.environment.get_spend()
    if not args.skip_saving:
        f.write(
            json.dumps(
                {
                    "input": agentversepipeline.environment.task_description,
                    "response": plan,
                    "logs": logs,
                    "decision_making_process": decision_making_process,
                    "spent": total_spent
                },  ensure_ascii=False
            )
            + "\n"
        )
        f.flush()
        f.close()
    return

if __name__ == "__main__":
    cli_main()
