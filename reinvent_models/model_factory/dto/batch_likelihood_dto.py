from dataclasses import dataclass
from typing import Union

import torch

from reinvent_models.mol2mol.dto.mol2mol_batch_dto import Mol2MolBatchDTO


@dataclass
class BatchLikelihoodDTO:
    batch: Union[Mol2MolBatchDTO]
    likelihood: torch.Tensor