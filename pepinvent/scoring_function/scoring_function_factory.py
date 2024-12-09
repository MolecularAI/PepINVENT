from pepinvent.scoring_function.geometric_mean import GeometricMean
from pepinvent.scoring_function.scoring_component_factory import ScoringComponentFactory
from pepinvent.scoring_function.scoring_config import ScoringConfig
from pepinvent.scoring_function.weighted_average import WeightedAverage


class ScoringFunctionFactory:
    def __init__(self, parameters: ScoringConfig):
        self._parameters = parameters
        self._component_factory = ScoringComponentFactory()
        self._registry = {"weighted_average": WeightedAverage,
                          "geometric_mean": GeometricMean,
                          }

    def create_scoring_function(self):
        scoring_function = self._registry.get(self._parameters.scoring_function)

        list_of_scoring_components = [self._component_factory.create_scoring_component(component)
                                      for component in self._parameters.scoring_components]
        instance = scoring_function(list_of_scoring_components, None)
        return instance
