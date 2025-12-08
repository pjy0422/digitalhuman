from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, List, Tuple

from pydantic import BaseModel

from agentverse.message import EvaluatorMessage

if TYPE_CHECKING:
    from agentverse.agents import EvaluatorFinalAgent
    from agentverse.message import EvaluatorMessage, SolverMessage, ExecutorMessage

from . import evaluator_final_registry


class BaseEvaluatorFinal(BaseModel):
    """
    The base class of execution.
    """

    @abstractmethod
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
        pass

    def reset(self):
        pass


@evaluator_final_registry.register("none")
class NoneEvaluatorFinal(BaseEvaluatorFinal):
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
        result = EvaluatorMessage(
            score=0, advice="\n".join([r.content for r in result])
        )
        return result


@evaluator_final_registry.register("dummy")
class DummyEvaluatorFinal(BaseEvaluatorFinal):
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
        result = EvaluatorMessage(score=1, advice="")
        return result


@evaluator_final_registry.register("dummy")
class DummyEvaluatorFinal(BaseEvaluatorFinal):
    async def astep(
        self,
        agent: EvaluatorFinalAgent,
        solution: List[str] | str,
        result: List[str] | str,
        task_description: str,
        all_role_description: List[str],
        *args,
        **kwargs,
    ) -> EvaluatorMessage:
        result = EvaluatorMessage(
            score=0, advice="\n".join([r.content for r in result])
        )
        return result
