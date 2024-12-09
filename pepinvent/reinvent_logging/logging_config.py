from pydantic.dataclasses import dataclass


@dataclass
class LoggingConfig:
    logging_path: str
    result_path: str
