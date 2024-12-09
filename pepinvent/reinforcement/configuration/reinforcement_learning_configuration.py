from pydantic import BaseModel, Field

from pepinvent.reinforcement.configuration.learning_config import LearningConfig
from pepinvent.reinforcement.diversity_filters.diversity_filter_parameters import DiversityFilterParameters
from pepinvent.reinvent_logging.logging_config import LoggingConfig
from pepinvent.scoring_function.scoring_config import ScoringConfig


class ReinforcementLearningConfiguration(BaseModel):
    name: str
    model_type: str
    model_path: str
    input_sequence: str
    learning_configuration: LearningConfig
    scoring_function: ScoringConfig
    logging: LoggingConfig
    diversity_filter: DiversityFilterParameters = Field(default_factory=DiversityFilterParameters)
    run_type: str = 'reinforcement'

