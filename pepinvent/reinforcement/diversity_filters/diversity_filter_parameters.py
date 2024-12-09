from dataclasses import dataclass


@dataclass
class DiversityFilterParameters:
    name: str = "NoFilter"
    score_threshold: float = 0.4
    bucket_size: int = 25
    similarity_threshold: float = 0.4
    penalty: float = 0.5
