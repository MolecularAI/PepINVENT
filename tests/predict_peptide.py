import os
import unittest

from pepinvent.reinforcement.dto.scoring_input_dto import ScoringInputDTO
from pepinvent.scoring_function.scoring_components.predictive_model import PredictiveModel
from pepinvent.scoring_function.scoring_components.scoring_component_parameters import ScoringComponentParameters


class TestPredictiveModel(unittest.TestCase):
    def setUp(self):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        peptides = ['N1(C)[C@@H]([C@H](O)[C@H](C)CC=CC)C(=O)N[C@@H](CC)C(=O)N(C)CC(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@H](C)C(=O)N[C@H](C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N(C)[C@@H](C(C)C)C1(=O)']
        peptide_input = '?|?|?|N(C)[C@@H](CC(C)C)C(=O)|N[C@@H](C(C)C)C(=O)|N(C)[C@@H](CC(C)C)C(=O)|N[C@@H](C)C(=O)|N[C@H](C)C(=O)|N(C)[C@@H](CC(C)C)C(=O)|N(C)[C@@H](CC(C)C)C(=O)|N(C)[C@@H](C(C)C)C1(=O)'
        peptide_output = ['N1(C)[C@@H]([C@H](O)[C@H](C)CC=CC)C(=O)|N[C@@H](CC)C(=O)|N(C)CC(=O)']
        self.score_input = ScoringInputDTO(peptides=peptides, peptide_input=peptide_input, peptide_outputs=peptide_output)

        params = ScoringComponentParameters(name='predictive_model', weight=1, specific_parameters={
            "transformation": {"transformation_type": "no_transformation"},
            "model_path": f"{ROOT_DIR}/../models/predictive_model.pckl",
        "scalar_path": f"{ROOT_DIR}/../models/feature_scalar.pckl"})
        self.scoring_function_1 = PredictiveModel(params)




    def test_prediction(self):
        final_summary = self.scoring_function_1.calculate_score(self.score_input)
        print(final_summary.total_score)
