from dataclasses import dataclass

from reinvent_models.mol2mol.models.vocabulary import Vocabulary


@dataclass
class Mol2MolNetworkParameters:
    vocabulary_size: int = 0
    num_layers: int = 6
    num_heads: int = 8
    model_dimension: int = 256
    feedforward_dimension: int = 2048
    dropout: float = 0.1


@dataclass
class Mol2MolModelParameterDTO:
    vocabulary: Vocabulary
    max_sequence_length: int
    network_parameter: Mol2MolNetworkParameters
    network_state: dict
