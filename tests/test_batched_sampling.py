import os
import unittest

from pepinvent.sampling.sampling import Sampling
from pepinvent.sampling.sampling_config import SamplingConfig
from reinvent_models.model_factory.configurations.model_configuration import ModelConfiguration
from reinvent_models.model_factory.enums.model_mode_enum import ModelModeEnum
from reinvent_models.model_factory.enums.model_type_enum import ModelTypeEnum
from reinvent_models.model_factory.generative_model import GenerativeModel
from reinvent_models.model_factory.mol2mol_adapter import Mol2MolAdapter




class TestBatchedSampling(unittest.TestCase):

    def setUp(self):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        model_type = ModelTypeEnum()
        model_mode = ModelModeEnum()
        model_file_path = f"{ROOT_DIR}/../models/generative_model.ckpt"
        model_config = ModelConfiguration(model_type=model_type.MOL2MOL, model_mode=model_mode.INFERENCE,
                                          model_file_path=model_file_path)
        self.agent: Mol2MolAdapter = GenerativeModel(model_config)

        self._sampling_config = SamplingConfig(model_type=model_type.MOL2MOL, model_path=model_file_path,
                                               results_output="",
                                               input_sequences_path=f"{ROOT_DIR}/fixtures/sampling_data.csv", num_samples=10)
        self._sampling = Sampling(self.agent, self._sampling_config)

    def test_likelihood(self):
        masked_peptides = self._sampling.load_data()
        dtos = self._sampling._sample(masked_peptides)
        print(dtos)