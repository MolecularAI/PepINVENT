from dataclasses import dataclass

import torch


@dataclass
class Mol2MolBatchDTO:
    input: torch.Tensor
    input_mask: torch.Tensor
    output: torch.Tensor
    output_mask: torch.Tensor
    tanimoto: torch.Tensor = None