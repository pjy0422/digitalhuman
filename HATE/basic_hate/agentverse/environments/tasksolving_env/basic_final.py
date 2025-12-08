import asyncio
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

from colorama import Fore

from agentverse.environments import BaseEnvironment
from agentverse.agents.base import BaseAgent
from agentverse.logging import logger
from agentverse.message import Message, SolverMessage, ExecutorMessage
from agentverse.utils import AGENT_TYPES

from .. import env_registry as EnvironmentRegistry

from agentverse.environments.tasksolving_env.rules import TasksolvingRuleFinal


@EnvironmentRegistry.register("task-basic-final")
class BasicEnvironmentFinal(BaseEnvironment):
    rule: TasksolvingRuleFinal
    agents: Dict[Enum, Union[BaseAgent, List[BaseAgent]]] = None

    task_description: str

    cnt_turn: int = 0
    max_turn: int = 10
    success: bool = False
    iter_samples: int = 3
    agents_score: list = []

    def __init__(self, **kwargs):
        rule_config = kwargs.pop("rule", {})
        role_assigner_config = rule_config.pop(
            "role_assigner", {"type": "role_description"}
        )
        decision_maker_config = rule_config.pop("decision_maker", {"type": "vertical"})
        executor_config = rule_config.pop("executor", {"type": "none"})
        evaluator_config = rule_config.pop("evaluator", {"type": "basic"})
        evaluator_final_config = rule_config.pop("evaluator_final", {"type": "basic"})
        rule = TasksolvingRuleFinal(
            role_assigner_config=role_assigner_config,
            decision_maker_config=decision_maker_config,
            executor_config=executor_config,
            evaluator_config=evaluator_config,
            evaluator_final_config=evaluator_final_config,
        )
        print(rule_config)
        iter_samples = rule_config.pop("iter_samples", 3) # benchmark length
        agents_score = rule_config.pop("agent_score_initial", [])
        print(agents_score)
        super().__init__(rule=rule, iter_samples=iter_samples, agents_score=agents_score, **kwargs)

    async def step(
        self, advice: str = "No advice yet.", previous_plan: str = "No solution yet."
    ) -> List[Message]:
        # for sample_id in range(self.iter_samples):
        result = ""
        logs = []

        logger.info(f"Loop Round {self.cnt_turn}")

        # ================== EXPERT RECRUITMENT ==================
        agents = await self.rule.role_assign(
            self.task_description, self.agents, self.cnt_turn, advice
        )
        description = "\n".join([agent.role_description for agent in agents])
        logs.append({"module": "Role Assigner", "content": description})
        logger.info("", f"Role Assignment:\n{description}", Fore.CYAN)
        # ================== EXPERT RECRUITMENT ==================

        # ================== DECISION MAKING ==================
        plan, decision_making_process = await self.rule.decision_making(
            self.task_description, self.agents, previous_plan, advice
        )
        flatten_plan = "\n".join([p.content for p in plan])
        logs.append({"module": "Decision Maker", "content": flatten_plan})
        logger.info("", f"Decision Plan:\n{flatten_plan}", Fore.YELLOW)
        # print('plan--', plan)
        # ================== DECISION MAKING ==================

        # ================== EXECUTION ==================
        result: List[ExecutorMessage] = await self.rule.execute(
            self.task_description, self.agents, plan
        )
        flatten_result = "\n".join([r.content for r in result])
        logs.append({"module": "Executor", "content": flatten_result})
        logger.info("", f"Execution Result:", Fore.GREEN)
        logger.info("", flatten_result, Fore.GREEN)
        # ================== EXECUTION ==================

        # ================== EVALUATION ==================
        # print('plan--', plan)
        # print(self.agents[AGENT_TYPES.EVALUATION])
        if AGENT_TYPES.EVALUATION in self.agents.keys():

            score, advice = await self.rule.evaluate(
                self.task_description, self.agents, plan, result
            )
            logs.append(
                {
                    "agent": "evaluator",
                    "content": f"Evaluation result: Score: {score}\nAdvice: {advice}",
                }
            )
            logger.info(
                "", f"Evaluation result:\nScore: {score}\nAdvice: {advice}", Fore.YELLOW
            )

            if score is not None and (
                (isinstance(score, bool) and score is True)
                or (isinstance(score, (list, tuple)) and all([s >= 8 for s in score]))
                or (isinstance(score, (int, float)) and score >= 10)
            ):
                # TODO: 8 is an arbitrary threshold
                logs.append({"agent": "system", "content": "Good score! Accept!"})
                logger.info(
                    "", f"Good score! Accept! Final Result:\n{flatten_plan}", Fore.GREEN
                )
                self.success = True
            else:
                logs.append({"agent": "system", "content": "Bad score! Reject!"})
                logger.info("", "Bad score! Reject!", Fore.RED)
        else:
            advice = None
        self.cnt_turn += 1

        if self.is_done():
            # ================== EVALUATION_FINAL ==================
        # print('plan--', plan)
        # print(self.agents[AGENT_TYPES.EVALUATION])
            if AGENT_TYPES.EVALUATION_FINAL in self.agents.keys():
                score, advice = await self.rule.evaluate_final(
                    self.task_description, self.agents, plan, result
                )
                logs.append(
                    {
                        "agent": "evaluator_final",
                        "content": f"Evaluation result: Score: {score}\nAdvice: {advice}",
                    }
                )
                logger.info(
                    "", f"Evaluation result:\nScore: {score}\nAdvice: {advice}", Fore.YELLOW
                )
                # if score is not None:
                #     self.change_agents_score(score)
                #     self.rm_agents()
                # else:
                self.change_agents_score_select(advice, rule=0)
                # self.rm_agents()
                
                decision_making_process.append('[evaluator_final]: '+advice)

        return flatten_result, advice, flatten_plan, logs, self.success, decision_making_process

    def iter_agents(self):
        for role, agent_or_agents in self.agents.items():
            if isinstance(agent_or_agents, list):
                for agent in agent_or_agents:
                    yield role, agent
            else:
                yield role, agent_or_agents

    def get_spend(self):
        total_spent = sum([agent.get_spend() for (_, agent) in self.iter_agents()])
        return total_spent

    def report_metrics(self) -> None:
        logger.info("", "Agent spend:", Fore.GREEN)
        for role, agent in self.iter_agents():
            name = agent.name.split(":")[0]
            logger.info(
                "",
                f"Agent (Role: {role}) {name}: {agent.get_spend_formatted()}",
                Fore.GREEN,
            )
        logger.info("", f"Total spent: ${self.get_spend():.6f}", Fore.GREEN)

    def is_done(self):
        """Check if the environment is done"""
        return self.cnt_turn >= self.max_turn or self.success

    def set_task_description(self, task_description: str = ""):
        self.task_description = task_description

    def reset(self) -> None:
        """Reset the environment"""
        self.cnt_turn = 0
        self.rule.reset()

    def change_agents_score(self, score: list):
        for i in range(len(self.agents[AGENT_TYPES.CRITIC])):
            self.agents_score[i] += score[i]
        
    def change_agents_score_select(self, advice: str, rule: int):
        print(advice)
        if rule == 0:
            for i in range(len(self.agents[AGENT_TYPES.CRITIC])): # TODO remove agent will make index wrong
                if 'Agent'+str(i+1) in advice:
                    self.agents[AGENT_TYPES.CRITIC][i].agent_score += 10
                else:
                    self.agents[AGENT_TYPES.CRITIC][i].agent_score -= 10
                print(i, '--', self.agents[AGENT_TYPES.CRITIC][i].agent_score)
        else:
            pass
    
    # def rm_agents(self):
    #     for i in range(len(self.agents[AGENT_TYPES.CRITIC])):
    #         if self.agents_score[i] < 0:
    #             self.agents[AGENT_TYPES.CRITIC].pop(i)
