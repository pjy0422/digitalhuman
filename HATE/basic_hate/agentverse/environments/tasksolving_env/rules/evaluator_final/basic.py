from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

from . import evaluator_final_registry
from .base import BaseEvaluatorFinal

if TYPE_CHECKING:
    from agentverse.agents import EvaluatorFinalAgent
    from agentverse.message import EvaluatorMessage, SolverMessage, ExecutorMessage


@evaluator_final_registry.register("basic")
class BasicEvaluatorFinal(BaseEvaluatorFinal):
    cnt_agents: int = 0

    async def astep(
        self,
        agent: EvaluatorFinalAgent,
        solution: List[SolverMessage],
        result: List[ExecutorMessage],
        task_description: str,
        all_role_description: List[str],
        *args,
        **kwargs,
    ) -> EvaluatorMessage:
        flatten_solution = "\n".join(['['+s.sender+']: '+  s.content for s in solution])
        flatten_result = "\n".join(['['+r.sender+']: '+r.content for r in result])
        flatten_all_role_description = "\n".join(all_role_description)
        evaluation = await agent.astep(
            flatten_solution,
            flatten_result,
            task_description,
            flatten_all_role_description,
        )
        return evaluation


@evaluator_final_registry.register("basic-message")
class BasicEvaluatorFinal(BaseEvaluatorFinal):
    cnt_agents: int = 0

    async def astep(
        self,
        agent: EvaluatorFinalAgent,
        solution: List[SolverMessage],
        result: List[ExecutorMessage],
        task_description: str,
        all_role_description: List[str],
        *args,
        **kwargs,
    ) -> EvaluatorMessage:
        flatten_solution = "\n".join([s.content for s in solution])
        flatten_result = "\n".join([r.content for r in result])
        flatten_all_role_description = "\n".join(all_role_description)
        agent.add_message_to_memory(result)
        evaluation = await agent.astep(
            flatten_solution,
            flatten_result,
            task_description,
            flatten_all_role_description,
        )
        agent.add_message_to_memory([evaluation])
        return evaluation
