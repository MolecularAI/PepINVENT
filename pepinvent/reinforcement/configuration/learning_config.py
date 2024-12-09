from pydantic.dataclasses import dataclass


@dataclass
class LearningConfig:
    number_steps: int = 200
    batch_size: int = 50
    learning_rate: float = 0.0001
    score_multiplier: int = 20
    distance_threshold: int = -20
