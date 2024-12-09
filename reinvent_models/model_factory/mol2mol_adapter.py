from typing import List

import torch
import torch.utils.data as tud

from reinvent_models.model_factory.dto.batch_likelihood_dto import BatchLikelihoodDTO
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_models.mol2mol.dataset.paired_dataset import PairedDataset
from reinvent_models.model_factory.dto.sampled_sequence_dto import SampledSequencesDTO
from reinvent_models.mol2mol.enums.sampling_mode_enum import SamplingModesEnum

from reinvent_models.mol2mol.mol2mol_model import Mol2MolModel


class Mol2MolAdapter(GenerativeModelBase):
    def __init__(self, path_to_file: str, mode: str):
        self.generative_model = Mol2MolModel.load_from_file(path_to_file, mode)
        self.vocabulary = self.generative_model.vocabulary
        self.max_sequence_length = self.generative_model.max_sequence_length
        self.network = self.generative_model.network
        self.tokenizer = self.generative_model.tokenizer
        self.device = self.generative_model.device

    def save_to_file(self, path):
        self.generative_model.save_to_file(path)

    def likelihood(self, src, src_mask, trg, trg_mask) -> torch.Tensor:
        return self.generative_model.likelihood(
            src.to(self.device),
            src_mask.to(self.device),
            trg.to(self.device),
            trg_mask.to(self.device)
        )

    def likelihood_smiles(
        self, sampled_sequence_list: List[SampledSequencesDTO]
    ) -> BatchLikelihoodDTO:
        input = [dto.input for dto in sampled_sequence_list]
        output = [dto.output for dto in sampled_sequence_list]
        dataset = PairedDataset(
            input, output, vocabulary=self.vocabulary, tokenizer=self.tokenizer
        )
        data_loader = tud.DataLoader(
            dataset, len(dataset), shuffle=False, collate_fn=PairedDataset.collate_fn
        )

        for batch in data_loader:
            likelihood = self.likelihood(
                batch.input.to(self.device),
                batch.input_mask.to(self.device),
                batch.output.to(self.device),
                batch.output_mask.to(self.device),
            )
            dto = BatchLikelihoodDTO(batch, likelihood)
            return dto

    def sample(
        self, src, src_mask, decode_type=SamplingModesEnum.MULTINOMIAL
    ) -> List[SampledSequencesDTO]:
        return self.generative_model.sample(
            src.to(self.device), src_mask.to(self.device), decode_type
        )

    def set_beam_size(self, beam_size: int):
        self.generative_model.set_beam_size(beam_size)

    def set_temperature(self, temperature: float=1.0):
        self.generative_model.set_temperature(temperature)

    def set_mode(self, mode: str):
        self.generative_model.set_mode(mode)

    def get_network_parameters(self):
        return self.generative_model.get_network_parameters()

    def get_vocabulary(self):
        return self.vocabulary
