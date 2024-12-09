from typing import List, Any

from pydantic.dataclasses import dataclass


@dataclass
class OutputDTO:
    peptide: Any
    amino_acids: List[str]
