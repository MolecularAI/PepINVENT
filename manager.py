from typing import Union

from rdkit import RDLogger

from reinvent_models.model_factory.configurations.model_configuration import ModelConfiguration
from reinvent_models.model_factory.enums.model_mode_enum import ModelModeEnum
from reinvent_models.model_factory.enums.model_type_enum import ModelTypeEnum
from reinvent_models.model_factory.generative_model import GenerativeModel

from pepinvent.reinforcement.configuration.reinforcement_learning_configuration import ReinforcementLearningConfiguration
from pepinvent.reinforcement.learning_scenario import LearningScenario
from pepinvent.sampling.sampling import Sampling
from pepinvent.sampling.sampling_config import SamplingConfig

RDLogger.DisableLog('rdApp.*')


class Manager:
    def __init__(self, configuration: Union[ReinforcementLearningConfiguration, SamplingConfig]):
        model_type = ModelTypeEnum()
        model_mode = ModelModeEnum()
        self.configuration = configuration
        agent_config = ModelConfiguration(model_type=model_type.MOL2MOL, model_mode=model_mode.INFERENCE,
                                          model_file_path=configuration.model_path)
        self.agent = GenerativeModel(agent_config)

        prior_config = ModelConfiguration(model_type=model_type.MOL2MOL, model_mode=model_mode.INFERENCE,
                                          model_file_path=configuration.model_path)
        self.prior = GenerativeModel(prior_config)

    def execute(self):
        self.suppress_rdkit_warnings()
        if self.configuration.run_type == "reinforcement":
            learning_scenario = LearningScenario(self.agent, self.prior, self.configuration)
        else:
            learning_scenario = Sampling(self.agent, self.configuration)
        learning_scenario.execute()

    def suppress_rdkit_warnings(self):
        RDLogger.DisableLog('rdApp.*')