from copy import deepcopy
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from pepinvent.reinforcement.diversity_filters.base_diversity_filter import BaseDiversityFilter
from pepinvent.reinforcement.diversity_filters.diversity_filter_parameters import DiversityFilterParameters
from reinvent_models.model_factory.dto.sampled_sequence_dto import SampledSequencesDTO

from pepinvent.scoring_function.score_summary import FinalSummary


class IdenticalMurckoScaffold(BaseDiversityFilter):
    """Penalizes compounds based on exact Murcko Scaffolds previously generated."""

    def __init__(self, parameters: DiversityFilterParameters):
        super().__init__(parameters)

    def update_score(self, score_summary: FinalSummary, sampled_sequeces: List[SampledSequencesDTO], step=0) -> np.array:
        score_summary = deepcopy(score_summary)
        scores = score_summary.total_score
        smiles = score_summary.scored_smiles

        for i, chuckles in enumerate(score_summary.scored_smiles):
            if self._chemistry.smile_to_mol(chuckles):  # Checks validity
                smile = self._chemistry.convert_to_rdkit_smiles(chuckles, sanitize=True, isomericSmiles=True)
                scaffold = self._calculate_scaffold(smile)
                scores[i] = 0 if self._smiles_exists(smile) else scores[i]
                # scores[i] = self.parameters.penalty * scores[i] if self._smiles_exists(smile) else scores[i]

                if scores[i] >= self.parameters.score_threshold:
                    self._add_to_memory(i, scores[i], smile, chuckles, score_summary.scaffold_log, step, scaffold)
                    scores[i] = self._penalize_score(scaffold, scores[i])

        return scores

    def _calculate_scaffold(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if mol:
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=False)
            except ValueError:
                scaffold_smiles = ''
        else:
            scaffold_smiles = ''
        return scaffold_smiles
