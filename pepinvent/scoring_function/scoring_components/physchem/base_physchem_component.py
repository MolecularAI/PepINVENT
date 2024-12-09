from abc import abstractmethod

import numpy as np
from rdkit import Chem
from reinvent_chemistry import PhysChemDescriptors

from pepinvent.reinforcement.dto.scoring_input_dto import ScoringInputDTO
from pepinvent.scoring_function.score_summary import ComponentSummary
from pepinvent.scoring_function.scoring_components.base_score_component import BaseScoreComponent
from pepinvent.scoring_function.scoring_components.scoring_component_parameters import ScoringComponentParameters


class BasePhysChemComponent(BaseScoreComponent):
    def __init__(self, parameters: ScoringComponentParameters):
        super().__init__(parameters)
        self.phys_chem_descriptors = PhysChemDescriptors()

    def calculate_score(self, scoring_input: ScoringInputDTO, step=-1) -> ComponentSummary:
        molecules = [Chem.MolFromSmiles(pep) for pep in scoring_input.peptides]
        score, raw_score = self._calculate_score(molecules)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters, raw_score=raw_score)
        return score_summary

    def _calculate_score(self, query_mols) -> np.array:
        scores = []
        for mol in query_mols:
            try:
                score = self._calculate_phys_chem_property(mol)
            except ValueError:
                score = 0.0
            scores.append(score)
        transform_params = self.parameters.specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION, {}
        )
        transformed_scores = self._transformation_function(scores, transform_params)
        return np.array(transformed_scores, dtype=np.float32), np.array(scores, dtype=np.float32)

    @abstractmethod
    def _calculate_phys_chem_property(self, mol):
        raise NotImplementedError("_calculate_phys_chem_property method is not implemented")
