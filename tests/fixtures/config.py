import json

from pydantic.dataclasses import dataclass


@dataclass
class TestInputDTO:
    model_path: str
    test_folder: str

def read_json_file(path):
    with open(path, 'r') as f:
        config = json.load(f)
    return config



