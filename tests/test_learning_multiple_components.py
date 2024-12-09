import os
import shutil
import unittest

from rdkit import RDLogger

from pepinvent.reinforcement.configuration.learning_config import LearningConfig
from pepinvent.reinforcement.configuration.reinforcement_learning_configuration import \
    ReinforcementLearningConfiguration
from pepinvent.reinforcement.diversity_filters.diversity_filter_parameters import DiversityFilterParameters
from pepinvent.reinforcement.learning_scenario import LearningScenario
from pepinvent.reinvent_logging.logging_config import LoggingConfig
from pepinvent.scoring_function.scoring_components.scoring_component_parameters import ScoringComponentParameters
from pepinvent.scoring_function.scoring_config import ScoringConfig
from reinvent_models.model_factory.configurations.model_configuration import ModelConfiguration
from reinvent_models.model_factory.enums.model_mode_enum import ModelModeEnum
from reinvent_models.model_factory.enums.model_type_enum import ModelTypeEnum
from reinvent_models.model_factory.generative_model import GenerativeModel

from reinvent_models.model_factory.mol2mol_adapter import Mol2MolAdapter
from tests.fixtures.config import TestInputDTO, read_json_file

RDLogger.DisableLog('rdApp.*')

class TestMultipleComponents(unittest.TestCase):

    def setUp(self):
        project_root = os.path.dirname(__file__)
        config = read_json_file(os.path.join(project_root, 'fixtures/test_config.json'))
        sampling_paths = TestInputDTO(**config)

        model_type = ModelTypeEnum()
        model_mode = ModelModeEnum()

        model_config = ModelConfiguration(model_type=model_type.MOL2MOL, model_mode=model_mode.INFERENCE,
                                          model_file_path=sampling_paths.model_path)
        model_config_prior = ModelConfiguration(model_type=model_type.MOL2MOL, model_mode=model_mode.INFERENCE,
                                          model_file_path=sampling_paths.model_path)

        model: Mol2MolAdapter = GenerativeModel(model_config)
        prior: Mol2MolAdapter = GenerativeModel(model_config_prior)

        self.logging_path = f'{sampling_paths.test_folder}/tensorboard'
        self.result_folder = f'{sampling_paths.test_folder}/results'

        logging_config = LoggingConfig(logging_path=self.logging_path, result_path=self.result_folder)
        learning_configuration = LearningConfig(number_steps=10, batch_size=32, score_multiplier=50)
        component_1 = ScoringComponentParameters(name='maximum_ring_size', weight=1, specific_parameters={"transformation" : {"transformation_type": "double_sigmoid", "low": 12, "high":40, "coef_div": 10, "coef_si":20, "coef_se":20}})
        component_2 = ScoringComponentParameters(name='substructure_match', weight=1, specific_parameters={'transformation': {'transformation_type': 'no_transformation'}, 'smiles': ['SS']})

        scoring_function = ScoringConfig(scoring_function='geometric_mean',
                                         scoring_components=[component_1, component_2])
        diversity_filter = DiversityFilterParameters(name='IdenticalMurckoScaffold')
        self.configuration = ReinforcementLearningConfiguration(name='multiple_components',
                                                                model_type=model_type.MOL2MOL,
                                                                model_path=sampling_paths.model_path,
                                                                input_sequence="?|?|N[C@@H](C)C(=O)|?|N[C@@H](C)C(=O)|N[C@@H](C)C(=O)|N[C@@H](C)C(=O)|N[C@@H](C)C(=O)|?",
                                                                learning_configuration=learning_configuration,
                                                                scoring_function=scoring_function,
                                                                logging=logging_config,
                                                                diversity_filter=diversity_filter)
        self.learning_scenario = LearningScenario(model, prior, self.configuration)


    def tearDown(self):
        shutil.rmtree(self.logging_path)
        shutil.rmtree(self.result_folder)


    def test_learning_multiple_components(self):
        self.learning_scenario.execute()
