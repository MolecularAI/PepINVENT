import abc
from typing import List

import numpy as np
import pandas as pd
from reinvent_chemistry import Conversions
from reinvent_models.model_factory.dto.sampled_sequence_dto import SampledSequencesDTO

from pepinvent.reinforcement.diversity_filters.diversity_filter_memory import DiversityFilterMemory
from pepinvent.reinforcement.diversity_filters.diversity_filter_parameters import DiversityFilterParameters
from pepinvent.scoring_function.score_summary import FinalSummary, ComponentSummary


class BaseDiversityFilter(abc.ABC):

    @abc.abstractmethod
    def __init__(self, parameters: DiversityFilterParameters):
        self.parameters = parameters
        self._diversity_filter_memory = DiversityFilterMemory()
        self._chemistry = Conversions()

    @abc.abstractmethod
    def update_score(self, score_summary: FinalSummary, sampled_sequeces: List[SampledSequencesDTO], step=0) -> np.array:
        raise NotImplementedError("The method 'update_score' is not implemented!")

    def get_memory_as_dataframe(self) -> pd.DataFrame:
        return self._diversity_filter_memory.get_memory()

    def set_memory_from_dataframe(self, memory: pd.DataFrame):
        self._diversity_filter_memory.set_memory(memory)

    def number_of_smiles_in_memory(self) -> int:
        return self._diversity_filter_memory.number_of_smiles()

    def update_bucket_size(self, bucket_size: int):
        self.parameters.bucket_size = bucket_size


    def _smiles_exists(self, smile):
        return self._diversity_filter_memory.smiles_exists(smile)

    def _add_to_memory(self, indx: int, score, smile, sampled_sequence, components: List[ComponentSummary], step, scaffold: str = ''):
        self._diversity_filter_memory.update(indx, score, smile, sampled_sequence, components, step, scaffold)

    def _penalize_score(self, scaffold, score):
        """Penalizes the score if the scaffold bucket is full"""
        if self._diversity_filter_memory.scaffold_instances_count(scaffold) > self.parameters.bucket_size:
            score = 0.
        return score
