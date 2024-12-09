import numpy as np
from rdkit import Chem

from pepinvent.reinforcement.dto.scoring_input_dto import ScoringInputDTO
from pepinvent.scoring_function.score_summary import ComponentSummary
from pepinvent.scoring_function.scoring_components.base_score_component import BaseScoreComponent
from pepinvent.scoring_function.scoring_components.scoring_component_parameters import ScoringComponentParameters


class MatchingSubstructure(BaseScoreComponent):
    def __init__(self, parameters: ScoringComponentParameters):
        super().__init__(parameters)
        self.target_smarts = self.parameters.specific_parameters.get('smiles', [])
        self._validate_inputs(self.target_smarts)

    def calculate_score(self, scoring_input: ScoringInputDTO, step=-1) -> ComponentSummary:
        molecules = [Chem.MolFromSmiles(pep) for pep in scoring_input.peptides]
        score = self._substructure_match(molecules, self.target_smarts)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters, raw_score=score)
        return score_summary

    def _substructure_match(self, query_mols, list_of_SMARTS):
        if len(list_of_SMARTS) == 0:
            return np.ones(len(query_mols), dtype=np.float32)

        match = [any([mol.HasSubstructMatch(Chem.MolFromSmarts(subst)) for subst in list_of_SMARTS
                      if Chem.MolFromSmarts(subst)]) for mol in query_mols]
        return 0.5 * (1 + np.array(match))

    def _validate_inputs(self, smiles):
        for smart in smiles:
            if Chem.MolFromSmarts(smart) is None:
                raise IOError(f"Invalid smarts pattern provided as a matching substructure: {smart}")