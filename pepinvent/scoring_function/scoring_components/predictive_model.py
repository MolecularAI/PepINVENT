import copy
from typing import List

import dill as pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from pepinvent.reinforcement.dto.scoring_input_dto import ScoringInputDTO
from pepinvent.scoring_function.score_summary import ComponentSummary
from pepinvent.scoring_function.scoring_components.base_score_component import BaseScoreComponent
from pepinvent.scoring_function.scoring_components.scoring_component_parameters import ScoringComponentParameters


class PredictiveModel(BaseScoreComponent):
    def __init__(self, parameters: ScoringComponentParameters):
        super().__init__(parameters)
        model_path = self.parameters.specific_parameters.get('model_path')
        scalar_path = self.parameters.specific_parameters.get('scalar_path')
        self._model = pickle.load(open(model_path, 'rb'))
        self._scalar = pickle.load(open(scalar_path, 'rb'))

    def calculate_score(self, molecules: ScoringInputDTO, step=-1) -> ComponentSummary:
        raw_scores = self._predict(molecules.peptides)
        scores = self._transformation(raw_scores)
        score_summary = ComponentSummary(total_score=scores, parameters=self.parameters, raw_score=raw_scores)
        return score_summary

    def _predict(self, generated_peptides: List[str]) -> np.ndarray:
        peptides = copy.copy(generated_peptides)

        x_test = self._calculate_descriptors(peptides)
        x_test = self._scalar.transform(x_test)

        predictions = self._model.predict_proba(x_test)
        scores = predictions[:, 1]
        return scores

    def _transformation(self, scores: np.ndarray) -> np.ndarray:
        transform_params = self.parameters.specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION, {}
        )
        transformed_scores = self._transformation_function(scores, transform_params)
        return np.array(transformed_scores)

    def _calculate_descriptors(self, molecules) -> np.ndarray:
        fingerprints = [self._generate_fingerprint(query) for query in list(molecules)]
        return np.asarray(fingerprints)

    def _generate_fingerprint(self, query_smiles):
        fps = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(query_smiles), radius=4, useChirality=True, useCounts=True)
        fps = self._convert_to_hashed(fps, 2048)
        return fps

    def _convert_to_hashed(self, fingerprint, size):
        arr = np.zeros((size,), np.int32)
        for idx, v in fingerprint.GetNonzeroElements().items():
            nidx = idx % size
            arr[nidx] += int(v)
        return arr
