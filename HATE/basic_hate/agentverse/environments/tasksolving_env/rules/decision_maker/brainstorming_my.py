from __future__ import annotations
import asyncio
from colorama import Fore

from typing import TYPE_CHECKING, List

from . import decision_maker_registry
from .base import BaseDecisionMaker
from agentverse.logging import logger

from agentverse.message import Message

if TYPE_CHECKING:
    from agentverse.agents.base import BaseAgent
    from agentverse.message import CriticMessage


@decision_maker_registry.register("brainstorming")
class BrainstormingDecisionMaker(BaseDecisionMaker):
    """
    Much like the horizontal decision maker, but with some twists:
    (1) Solver acts as a summarizer, summarizing the discussion of this turn
    (2) After summarizing, all the agents' memory are cleared, and replaced with
    the summary (to avoid exceeding maximum context length of the model too fast)
    """

    name: str = "brainstorming"

    async def astep(
        self,
        agents: List[BaseAgent],
        task_description: str,
        previous_plan: str = "No solution yet.",
        advice: str = "No advice yet.",
        *args,
        **kwargs,
    ) -> List[str]:
        results = []
        decision_making_process = []
        if advice != "No advice yet.":
            self.broadcast_messages(
                agents, [Message(content=advice, sender="Evaluator")]
            )
        # for agent in agents[1:]:
        for agent in agents:
            review: CriticMessage = await agent.astep(
                previous_plan, advice, task_description
            )
            results.append(review)
            decision_making_process.append(f"[{review.sender}]: {review.content}")
            # if review.content != "":
            #     self.broadcast_messages(agents, [review])

            logger.info("", "Reviews:", Fore.YELLOW)
            logger.info(
                "",
                f"[{review.sender}]: {review.content}",
                Fore.YELLOW,
            )
        for review in results:
            if review.content != "":
                self.broadcast_messages(agents, [review])
        # result = await agents[0].astep(previous_plan, advice, task_description)
        # decision_making_process.append(f"[{result.sender}]: {result.content}")
        # for agent in agents:
        #     agent.memory.reset()
        # self.broadcast_messages(
        #     agents,
        #     [
        #         Message(
        #             content=result.content, sender="Summary From Previous Discussion Round"
        #         )
        #     ],
        # )
        return results, decision_making_process
