from typing import List

from pydantic.dataclasses import dataclass


@dataclass
class ScoringInputDTO:
    peptides: List[str]
    peptide_input: str = None
    peptide_outputs: List[str] = None
