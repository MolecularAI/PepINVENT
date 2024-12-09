import logging
import os
from abc import ABC, abstractmethod

import torch
from reinvent_scoring.scoring.enums.component_specific_parameters_enum import ComponentSpecificParametersEnum

from pepinvent.reinforcement.dto.output_dto import OutputDTO
from pepinvent.reinvent_logging.logging_config import LoggingConfig
from pepinvent.scoring_function.score_summary import FinalSummary


class BaseReinforcementLogger(ABC):
    def __init__(self, rl_config: LoggingConfig):
        self._log_config = rl_config
        self._setup_workfolder()
        self._logger = self._setup_logger()
        self._specific_parameters_enum = ComponentSpecificParametersEnum()

    @abstractmethod
    def log_message(self, message: str):
        raise NotImplementedError("log_message method is not implemented")

    @abstractmethod
    def timestep_report(self, start_time, n_steps, step, score_summary: FinalSummary,
                        agent_likelihood: torch.tensor, prior_likelihood: torch.tensor,
                        augmented_likelihood: torch.tensor, report: OutputDTO):
        raise NotImplementedError("timestep_report method is not implemented")

    @abstractmethod
    def save_final_state(self, peptides):
        raise NotImplementedError("save_final_state method is not implemented")

    def _setup_workfolder(self):
        if not os.path.isdir(self._log_config.logging_path):
            os.makedirs(self._log_config.logging_path)
        if not os.path.isdir(self._log_config.result_path):
            os.makedirs(self._log_config.result_path)

    def _setup_logger(self):
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s: %(module)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger = logging.getLogger("reinforcement_logger")
        if not logger.handlers:
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger
