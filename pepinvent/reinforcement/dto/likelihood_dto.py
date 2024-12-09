from typing import Any

from pydantic.dataclasses import dataclass


@dataclass
class LikelihoodDTO:
    prior_likelihood: Any
    agent_likelihood: Any
    augmented_likelihood: Any
