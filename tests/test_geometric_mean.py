import unittest

from pepinvent.reinforcement.dto.scoring_input_dto import ScoringInputDTO
from pepinvent.scoring_function.scoring_components.scoring_component_parameters import ScoringComponentParameters
from pepinvent.scoring_function.scoring_config import ScoringConfig
from pepinvent.scoring_function.scoring_function_factory import ScoringFunctionFactory


class TestGeometricMean(unittest.TestCase):

    def setUp(self):
        component_1 = ScoringComponentParameters(name='molecular_weight', weight=1, specific_parameters={
            'transformation': {'transformation_type': 'sigmoid', 'low': 500, 'high': 1600, 'k': 0.25}})
        component_2 = ScoringComponentParameters(name='maximum_ring_size', weight=1, specific_parameters={
            'transformation': {'transformation_type': 'sigmoid', 'low': 12, 'high': 50, 'k': 0.3}})

        scoring_function = ScoringConfig(scoring_function='geometric_mean',
                                         scoring_components=[component_1, component_2])

        self.geometric_mean = ScoringFunctionFactory(scoring_function).create_scoring_function()

        self.scoring_input = ScoringInputDTO(peptides=['N2[C@@H](Cc1ccc(O)cc1)C(=O)N1[C@@H](CCC1)C(=O)N(C)[C@@H](CS)C(=O)N[C@@H](C(=O)N)C(=O)N[C@@H](CO)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@H](CN(C)S(=O)(=O)c1c[nH]c(=O)cc1)C(=O)N[C@@H](Cc1ccc(O)cc1)C2(=O)',
                                                       'INVALID',
                                                       'N2[C@@H](Cc1ccc(O)cc1)C(=O)N1[C@@H](CCC1)C(=O)N(C)[C@@H](CO)C(=O)N[C@@H](CCN)C(=O)N[C@@H](CO)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](C(=O)Nc1n(C)ncc1C#N)C(=O)N[C@@H](Cc1ccc(O)cc1)C2(=O)',
                                                       'N2[C@@H](Cc1ccc(O)cc1)C(=O)N1[C@@H](CCC1)C(=O)N(C)[C@@H](CO)C(=O)N[C@@H](CCN)C(=O)N[C@@H](CO)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](C(=O)Nc1n(C)ncc1C#N)C(=O)N[C@@H](Cc1ccc(O)cc1)C2(=O)',
                                                       'INVALID'],
                                             peptide_input='N2[C@@H](Cc1ccc(O)cc1)C(=O)|N1[C@@H](CCC1)C(=O)|?|?|N[C@@H](CO)C(=O)|N[C@@H](Cc1ccc(O)cc1)C(=O)|N[C@@H](CCCNC(=N)N)C(=O)|?|?',
                                             peptide_outputs=['peptide1', 'peptide2', 'peptide3', 'peptide4', 'peptide5'])

    def tearDown(self):
        pass

    def test_geometric_mean(self):
        result = self.geometric_mean.calculate_score(self.scoring_input)
        print(result.total_score)
