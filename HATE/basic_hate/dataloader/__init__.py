from agentverse.registry import Registry

dataloader_registry = Registry(name="dataloader")

from .gsm8k import GSM8KLoader
from .responsegen import ResponseGenLoader
from .humaneval import HumanevalLoader
from .commongen import CommongenLoader
from .creative_writing import Creative_writingLoader
from .mgsm import MGSMLoader
from .logic_grid import LogicGridLoader
from .mmlu_pro import MMLULoader
from .debate_covid import Debate_covidLoader
from .travelplanner import TravelplannerLoader
from .inductive import InductiveLoader
from .bbh import BBHLoader
from .cmt import cmtLoader
from .cqa import cqaLoader
from .hle_text import hle_textLoader
from .debate_persuasion import Debate_persuasion
from .research_questions import Research_questions
from .browsecomp import Browsecomp
