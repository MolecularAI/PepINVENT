from dataclasses import dataclass
from typing import Dict


@dataclass
class ScoringComponentParameters:
    name: str
    weight: int
    specific_parameters: Dict
