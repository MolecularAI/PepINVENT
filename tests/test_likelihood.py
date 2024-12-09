import os
import unittest

from reinvent_models.model_factory.configurations.model_configuration import ModelConfiguration
from reinvent_models.model_factory.dto.sampled_sequence_dto import SampledSequencesDTO
from reinvent_models.model_factory.enums.model_mode_enum import ModelModeEnum

from reinvent_models.model_factory.enums.model_type_enum import ModelTypeEnum
from reinvent_models.model_factory.generative_model import GenerativeModel
from reinvent_models.model_factory.mol2mol_adapter import Mol2MolAdapter


class TestSampling(unittest.TestCase):

    def setUp(self):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        model_type = ModelTypeEnum()
        model_mode = ModelModeEnum()
        model_config = ModelConfiguration(model_type=model_type.MOL2MOL, model_mode=model_mode.INFERENCE,
                                          model_file_path=f'{ROOT_DIR}/../models/generative_model.ckpt')
        model_config_prior = ModelConfiguration(model_type=model_type.MOL2MOL, model_mode=model_mode.INFERENCE,
                                                model_file_path=f'{ROOT_DIR}/../models/generative_model.ckpt')
        self.agent: Mol2MolAdapter = GenerativeModel(model_config)
        self.prior: Mol2MolAdapter = GenerativeModel(model_config_prior)

        input_smi = '?|?|?|N[C@@H](CC(=O)O)C(=O)|N[C@@H](CC(=O)O)C(=O)|N[C@@H](CC(=O)O)C(=O)|N[C@@H](CC(=O)O)C(=O)|N[C@@H](CC(=O)O)C(=O)|N[C@@H](CC(=O)O)C(=O)O'
        output_smi = 'N[C@H](CCC(=O)O)C(=O)|N(C)[C@@H](Cc1ccc(O)cc1)C(=O)|N(C)[C@@H](c1ccsc1Nc1c(Cl)cccc1Cl)C(=O)'
        self.model_input = [SampledSequencesDTO(input_smi, output_smi, 0)]



    def test_likelihood(self):
        agent_likelihood = -self.agent.likelihood_smiles(self.model_input).likelihood
        prior_likelihood = -self.prior.likelihood_smiles(self.model_input).likelihood
        self.assertEqual(agent_likelihood[0], prior_likelihood[0])
