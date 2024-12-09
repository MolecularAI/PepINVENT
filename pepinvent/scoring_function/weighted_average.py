import copy
from typing import List

import numpy as np

from pepinvent.reinforcement.chemistry import Chemistry
from pepinvent.reinforcement.dto.scoring_input_dto import ScoringInputDTO
from pepinvent.scoring_function.score_summary import ComponentSummary, FinalSummary
from pepinvent.scoring_function.scoring_components.base_score_component import BaseScoreComponent


class WeightedAverage:
    def __init__(self, scoring_components: List[BaseScoreComponent], logger):
        self._logger = logger
        self._scoring_components = scoring_components
        self._chemistry = Chemistry()

    def calculate_score(self, scoring_input: ScoringInputDTO) -> [np.ndarray, List]:
        peptides = copy.copy(scoring_input.peptides)
        molecules, valid_indices = self._chemistry.check_validity(peptides)
        scoring_input = self._strip_invalids(scoring_input, valid_indices)
        dtos = []
        for component in self._scoring_components:
            dto: ComponentSummary = component.calculate_score(scoring_input)
            scores = np.zeros(len(peptides))
            raw_scores = np.zeros(len(peptides))
            for idx, score_value, raw_value in zip(valid_indices, dto.total_score, dto.raw_score):
                scores[idx] = score_value
                raw_scores[idx] = raw_value
            dto.total_score = scores
            dto.raw_score = raw_scores
            dtos.append(dto)

        total_score = self._calculate_average(dtos, len(peptides))
        #
        peptides = [f'Invalid_{pep}' if idx not in valid_indices else pep for idx, pep in enumerate(peptides)]
        final_summary = FinalSummary(total_score=total_score,
                            scored_smiles=peptides,
                            valid_idxs=valid_indices,
                            scaffold_log_summary=dtos)
        return final_summary

    def _calculate_average(self, dtos: List[ComponentSummary], num_molecules) -> np.ndarray:
        substructure_dto = [dto for dto in dtos if dto.parameters.name == 'substructure_match']
        dtos = [dto for dto in dtos if dto.parameters.name != 'substructure_match']

        total_score = np.zeros((num_molecules, ))
        total_weight = .0
        for dto in dtos:
            total_score += dto.total_score * dto.parameters.weight
            total_weight += dto.parameters.weight
        final_score = total_score/total_weight
        if len(substructure_dto)==1:
            final_score = final_score * substructure_dto[0].total_score
        return final_score

    def _strip_invalids(self, scoring_input: ScoringInputDTO, valid_indices) -> ScoringInputDTO:
        scoring_input.peptide_outputs = [scoring_input.peptide_outputs[idx] for idx in valid_indices]
        scoring_input.peptides = [scoring_input.peptides[idx] for idx in valid_indices]
        return scoring_input


