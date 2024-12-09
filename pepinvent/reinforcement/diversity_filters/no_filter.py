from copy import deepcopy
from typing import List

import numpy as np

from pepinvent.reinforcement.diversity_filters.base_diversity_filter import BaseDiversityFilter
from pepinvent.reinforcement.diversity_filters.diversity_filter_parameters import DiversityFilterParameters
from reinvent_models.model_factory.dto.sampled_sequence_dto import SampledSequencesDTO
from pepinvent.scoring_function.score_summary import FinalSummary


class NoFilter(BaseDiversityFilter):
    """Don't penalize compounds."""

    def __init__(self, parameters: DiversityFilterParameters):
        super().__init__(parameters)

    def update_score(self, score_summary: FinalSummary, sampled_sequeces: List[SampledSequencesDTO], step=0) -> np.array:
        score_summary = deepcopy(score_summary)
        scores = score_summary.total_score
        smiles = score_summary.scored_smiles
        for i in score_summary.valid_idxs:
            if scores[i] >= self.parameters.score_threshold:
                smile = self._chemistry.convert_to_rdkit_smiles(smiles[i])
                self._add_to_memory(i, scores[i], smile, sampled_sequeces[i].output, score_summary.scaffold_log, step)
        return scores