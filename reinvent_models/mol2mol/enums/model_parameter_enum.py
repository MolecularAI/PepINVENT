from dataclasses import dataclass


@dataclass(frozen=True)
class ModelParametersEnum:
    NUMBER_OF_LAYERS = "num_layers"
    NUMBER_OF_HEADS = "num_heads"
    MODEL_DIMENSION = 'model_dimension'
    FEED_FORWARD_DIMENSION = 'feedforward_dimension'
    VOCABULARY_SIZE = "vocabulary_size"
    DROPOUT = "dropout"
