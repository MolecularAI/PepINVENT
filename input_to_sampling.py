import json
from sys import argv

from manager import Manager
from pepinvent.sampling.sampling_config import SamplingConfig


def read_json_file(path):
    with open(path) as f:
        json_input = f.read().replace('\r', '').replace('\n', '')
    try:
        return json.loads(json_input)
    except (ValueError, KeyError, TypeError) as e:
        print(f"JSON format error in file ${path}: \n ${e}")

if __name__ == "__main__":
    path = argv[1]

    config = read_json_file(path)
    sampling_parameters = SamplingConfig.parse_obj(config)
    manager = Manager(sampling_parameters)
    manager.execute()
