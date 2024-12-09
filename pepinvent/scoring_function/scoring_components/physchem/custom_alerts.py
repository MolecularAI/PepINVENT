import numpy as np
from rdkit import Chem

from pepinvent.reinforcement.dto.scoring_input_dto import ScoringInputDTO
from pepinvent.scoring_function.score_summary import ComponentSummary
from pepinvent.scoring_function.scoring_components.base_score_component import BaseScoreComponent
from pepinvent.scoring_function.scoring_components.scoring_component_parameters import ScoringComponentParameters


class CustomAlerts(BaseScoreComponent):
    def __init__(self, parameters: ScoringComponentParameters):
        super().__init__(parameters)
        self.custom_alerts = self.parameters.specific_parameters.get('smarts', [])

    def calculate_score(self, scoring_input: ScoringInputDTO) -> ComponentSummary:
        score, raw_score = self.calculate_substructure_scores(scoring_input.peptides, self.custom_alerts)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters, raw_score=raw_score)
        return score_summary

    def calculate_substructure_scores(self, peptides, custom_alerts):
        molecules = [Chem.MolFromSmiles(pep) for pep in peptides]
        scores = self._substructure_match(molecules, custom_alerts)
        transform_params = self.parameters.specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION, {}
        )
        transformed_scores = self._transformation_function(scores, transform_params)
        return np.array(transformed_scores, dtype=np.float32), np.array(scores, dtype=np.float32)

    def _substructure_match(self, query_mols, list_of_SMARTS):
        match = [any([mol.HasSubstructMatch(Chem.MolFromSmarts(subst)) for subst in list_of_SMARTS
                      if Chem.MolFromSmarts(subst)]) for mol in query_mols]
        reverse = [1 - m for m in match]
        return reverse
