from agentverse.registry import Registry

evaluator_final_registry = Registry(name="EvaluatorFinalRegistry")

from .base import BaseEvaluatorFinal, NoneEvaluatorFinal
from .basic import BasicEvaluatorFinal
