from dataclasses import dataclass
from typing import List

import numpy as np

from pepinvent.scoring_function.scoring_components.scoring_component_parameters import ScoringComponentParameters


@dataclass
class ComponentSummary:
    total_score: np.ndarray
    parameters: ScoringComponentParameters
    raw_score: np.ndarray = None


class FinalSummary:
    def __init__(self, total_score: np.array, scored_smiles: List[str], valid_idxs: np.ndarray,
                 scaffold_log_summary: List[ComponentSummary]):
        self.total_score = total_score
        self.scored_smiles = scored_smiles
        self.valid_idxs = valid_idxs
        score = [LoggableComponent(c.parameters.name, c.parameters.name, c.total_score) for c in scaffold_log_summary]
        raw_score = [LoggableComponent(c.parameters.name, f'raw_{c.parameters.name}', c.raw_score) for c in
                     scaffold_log_summary if c.raw_score is not None]
        score.extend(raw_score)
        self.scaffold_log: List[ComponentSummary] = scaffold_log_summary
        self.profile: List[LoggableComponent] = score




@dataclass
class LoggableComponent:
    component_type: str
    name: str
    score: np.array