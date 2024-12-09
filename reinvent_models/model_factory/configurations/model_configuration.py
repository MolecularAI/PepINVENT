from dataclasses import dataclass


@dataclass
class ModelConfiguration:
    model_type: str
    model_mode: str
    model_file_path: str