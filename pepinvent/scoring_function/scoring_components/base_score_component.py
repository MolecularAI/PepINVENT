from abc import abstractmethod, ABC

from reinvent_scoring import ComponentSpecificParametersEnum
from reinvent_scoring.scoring.enums.transformation_parameters_enum import TransformationParametersEnum
from reinvent_scoring.scoring.enums.transformation_type_enum import TransformationTypeEnum
from reinvent_scoring.scoring.score_transformations import TransformationFactory

from pepinvent.reinforcement.dto.scoring_input_dto import ScoringInputDTO
from pepinvent.scoring_function.score_summary import ComponentSummary
from pepinvent.scoring_function.scoring_components.scoring_component_parameters import ScoringComponentParameters


class BaseScoreComponent(ABC):

    def __init__(self, parameters: ScoringComponentParameters):
        self.component_specific_parameters = ComponentSpecificParametersEnum()
        self.parameters = parameters
        self._transformation_function = self._assign_transformation()

    @abstractmethod
    def calculate_score(self, molecules: ScoringInputDTO, step=-1) -> ComponentSummary:
        raise NotImplementedError("calculate_score method is not implemented")

    def calculate_score_for_step(self, molecules: ScoringInputDTO, step=-1) -> ComponentSummary:
        return self.calculate_score(molecules)

    def _assign_transformation(self):
        transformation_type = TransformationTypeEnum()
        factory = TransformationFactory()
        transform_params = self.parameters.specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION, {})
        if transform_params:
            transform_function = factory.get_transformation_function(transform_params)
        else:
            self.parameters.specific_parameters[
                self.component_specific_parameters.TRANSFORMATION] = {
                    TransformationParametersEnum.TRANSFORMATION_TYPE: transformation_type.NO_TRANSFORMATION
                }
            transform_function = factory.no_transformation
        return transform_function
