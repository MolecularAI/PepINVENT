from reinvent_models.model_factory.configurations.model_configuration import ModelConfiguration
from reinvent_models.model_factory.enums.model_type_enum import ModelTypeEnum
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_models.model_factory.mol2mol_adapter import Mol2MolAdapter


class GenerativeModel:
    def __new__(cls, configuration: ModelConfiguration) -> GenerativeModelBase:
        cls._configuration = configuration
        model_type_enum = ModelTypeEnum()

        if cls._configuration.model_type == model_type_enum.MOL2MOL:
            model = Mol2MolAdapter(cls._configuration.model_file_path, mode=cls._configuration.model_mode)
        else:
            raise ValueError(f"Invalid model_type provided: '{cls._configuration.model_type}")
        return model

