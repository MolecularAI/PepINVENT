import json
from sys import argv

from manager import Manager
from pepinvent.reinforcement.configuration.reinforcement_learning_configuration import \
    ReinforcementLearningConfiguration


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
    learning_parameters = ReinforcementLearningConfiguration.parse_obj(config)
    manager = Manager(learning_parameters)
    manager.execute()
