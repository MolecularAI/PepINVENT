from typing import List
from pydantic.dataclasses import dataclass
from pepinvent.scoring_function.scoring_components.scoring_component_parameters import ScoringComponentParameters


@dataclass
class ScoringConfig:
    scoring_function: str
    scoring_components: List[ScoringComponentParameters]