from dataclasses import dataclass


@dataclass(frozen=True)
class ModelTypeEnum:
    MOL2MOL = "mol2mol"