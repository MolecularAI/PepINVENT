import copy
from typing import List

import numpy as np

from pepinvent.reinforcement.chemistry import Chemistry
from pepinvent.reinforcement.dto.scoring_input_dto import ScoringInputDTO
from pepinvent.reinvent_logging.local_reinforcement_logger import LocalReinforcementLogger
from pepinvent.scoring_function.score_summary import ComponentSummary, FinalSummary
from pepinvent.scoring_function.scoring_components.base_score_component import BaseScoreComponent


class GeometricMean:
    def __init__(self, scoring_components: List[BaseScoreComponent], logger: LocalReinforcementLogger):
        self._logger = logger
        self._scoring_components = scoring_components
        self._chemistry = Chemistry()


    def calculate_score(self, scoring_input: ScoringInputDTO) -> FinalSummary:

        # validity check
        cloned_scoring_input = copy.deepcopy(scoring_input)
        molecules, valid_indices = self._chemistry.check_validity(cloned_scoring_input.peptides)
        cloned_scoring_input = self._strip_invalids(cloned_scoring_input, valid_indices)

        dtos = []
        for component in self._scoring_components:
            dto: ComponentSummary = component.calculate_score(cloned_scoring_input)
            scores = np.zeros(len(scoring_input.peptides))
            raw_scores = np.zeros(len(scoring_input.peptides))
            for idx, score_value, raw_value in zip(valid_indices, dto.total_score, dto.raw_score):
                scores[idx] = score_value
                raw_scores[idx] = raw_value
            dto.total_score = scores
            dto.raw_score = raw_scores
            dtos.append(dto)

        total_score = self._calculate_geometric_mean(dtos, len(scoring_input.peptides))

        peptides = [f'Invalid_{pep}' if idx not in valid_indices else pep for idx, pep in enumerate(scoring_input.peptides)]
        final_summary = FinalSummary(total_score=total_score,
                            scored_smiles=peptides,
                            valid_idxs=valid_indices,
                            scaffold_log_summary=dtos)
        return final_summary

    def _calculate_geometric_mean(self, dtos: List[ComponentSummary], num_molecules) -> np.ndarray:
        substructure_dto = [dto for dto in dtos if dto.parameters.name == 'substructure_match']
        dtos = [dto for dto in dtos if dto.parameters.name != 'substructure_match']

        total_score = np.ones((num_molecules, ))
        total_weight = .0
        for dto in dtos:
            total_score *= dto.total_score ** dto.parameters.weight
            total_weight += dto.parameters.weight
        final_score = total_score**(1/total_weight)
        if len(substructure_dto)==1:
            final_score = final_score * substructure_dto[0].total_score
        return final_score

    def _strip_invalids(self, scoring_input: ScoringInputDTO, valid_indices) -> ScoringInputDTO:
        scoring_input.peptide_outputs = [scoring_input.peptide_outputs[idx] for idx in valid_indices]
        scoring_input.peptides = [scoring_input.peptides[idx] for idx in valid_indices]
        return scoring_input

    def weighted_geometric_mean(values, weights):
        weighted_values = np.power(values, weights)
        product = np.prod(weighted_values)
        sum_weights = np.sum(weights)
        return product ** (1 / sum_weights)


