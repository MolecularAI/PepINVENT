from typing import List, Tuple

from tqdm import tqdm
from torch import Tensor
from torch.autograd import Variable
from torch.utils import data as tud
import numpy as np
import torch

from reinvent_models.mol2mol.dto.mol2mol_batch_dto import Mol2MolBatchDTO
from reinvent_models.mol2mol.models.module.subsequent_mask import subsequent_mask
from reinvent_chemistry.conversions import Conversions
from reinvent_chemistry.similarity import Similarity

DEVICE='cpu'

class PairedDataset(tud.Dataset):
    """Dataset that takes a list of (input, output) pairs."""

    # TODO check None for en_input, en_output
    def __init__(
        self, smiles_input: List[str], smiles_output: List[str], vocabulary, tokenizer
    ):
        self.vocabulary = vocabulary
        self._tokenizer = tokenizer

        self._encoded_input_list = []
        self._encoded_output_list = []
        for input_smi, output_smi in zip(smiles_input, smiles_output):
            ok_input, ok_output = True, True
            try: 
                tokenized_input = self._tokenizer.tokenize(input_smi)
                en_input = self.vocabulary.encode(tokenized_input)
            except KeyError as e:
                print(f"WARNING. Input smile {input_smi} contains an invalid token {e}. It will be ignored")
                ok_input = False
            try:
                tokenized_output = self._tokenizer.tokenize(output_smi)
                en_output = self.vocabulary.encode(tokenized_output)
            except KeyError as e:
                print(f"WARNING. Output smile {output_smi} contains an invalid token {e}. It will be ignored")
                ok_output = False
            if ok_input and ok_output:
                self._encoded_input_list.append(en_input)
                self._encoded_output_list.append(en_output)

    def __getitem__(self, i):
        en_input, en_output = self._encoded_input_list[i], self._encoded_output_list[i]
        return (
            torch.tensor(en_input, dtype=torch.long, device=DEVICE),
            torch.tensor(en_output, dtype=torch.long, device=DEVICE),
        )  # pylint: disable=E1102

    def __len__(self):
        return len(self._encoded_input_list)

    @staticmethod
    def collate_fn(encoded_pairs) -> Mol2MolBatchDTO:
        """Turns a list of encoded pairs (input, target) of sequences and turns them into two batches.

        :param: A list of pairs of encoded sequences.
        :return: A tuple with two tensors, one for the input and one for the targets in the same order as given.
        """

        encoded_inputs, encoded_targets = list(zip(*encoded_pairs))
        collated_arr_source, src_mask = _mask_batch(encoded_inputs)
        collated_arr_target, trg_mask = _mask_batch(encoded_targets)

        # TODO: refactor the logic below
        trg_mask = trg_mask & Variable(
            subsequent_mask(collated_arr_target.size(-1)).type_as(trg_mask)
        )
        trg_mask = trg_mask[:, :-1, :-1]  # save start token, skip end token

        dto = Mol2MolBatchDTO(
            collated_arr_source,
            src_mask,
            collated_arr_target,
            trg_mask,
        )
        return dto


class StratifiedPairedDataset(tud.Dataset):
    """Dataset that takes a list of (input, output) pairs."""

    def __init__(
        self,
        smiles_input: List[str],
        smiles_output: List[str],
        vocabulary,
        tokenizer,
        target_per_source=4,
    ):
        self.vocabulary = vocabulary
        self._tokenizer = tokenizer
        self.target_per_source = target_per_source

        self.source_to_targets = {}
        self.smile_to_vec = {}
        self.idx_to_source = {}
        self.tanimoto_similarities = {}

        conversions = Conversions()
        similarity = Similarity()

        molecules = {}

        for input_smi, output_smi in tqdm(
            zip(smiles_input, smiles_output), total=len(smiles_input)
        ):
            try:

                if input_smi not in self.source_to_targets:
                    self.idx_to_source[len(self.source_to_targets)] = input_smi
                    self.source_to_targets[input_smi] = []

                if input_smi not in self.smile_to_vec:
                    tokenized_input = self._tokenizer.tokenize(input_smi)
                    en_input = self.vocabulary.encode(tokenized_input)
                    self.smile_to_vec[input_smi] = en_input

                if output_smi not in self.smile_to_vec:
                    tokenized_output = self._tokenizer.tokenize(output_smi)
                    en_output = self.vocabulary.encode(tokenized_output)
                    self.smile_to_vec[output_smi] = en_output

                if input_smi in molecules:
                    input_mol = molecules[input_smi]
                else:
                    input_mol = conversions.smiles_to_fingerprints([input_smi])
                    input_mol = input_mol[0] if len(input_mol) else None
                    molecules[input_smi] = input_mol

                if output_smi in molecules:
                    output_mol = molecules[output_smi]
                else:
                    output_mol = conversions.smiles_to_fingerprints([output_smi])
                    output_mol = output_mol[0] if len(output_mol) else None
                    molecules[output_smi] = output_mol

                if input_mol and output_mol:
                    ts = similarity.calculate_tanimoto([input_mol], [output_mol])
                    ts = float(ts.ravel())
                else:
                    ts = 0.0

                self.tanimoto_similarities[(input_smi, output_smi)] = ts
                self.source_to_targets[input_smi].append(output_smi)

            except KeyError as e:
                print(
                    f"Token {e} not found in the vocabulary. Pair ({input_smi}, {output_smi}) ignored."
                )

    def __getitem__(self, i):
        input_smile = self.idx_to_source[i]
        en_input = self.smile_to_vec[input_smile]

        idx = np.random.choice(
            len(self.source_to_targets[input_smile]),
            self.target_per_source,
            replace=(len(self.source_to_targets[input_smile]) < self.target_per_source),
        )
        output_smiles = self.source_to_targets[input_smile]

        return (
            [torch.tensor(en_input, dtype=torch.long, device=DEVICE) for _ in idx],
            [
                torch.tensor(self.smile_to_vec[output_smiles[eo]], dtype=torch.long, device=DEVICE)
                for eo in idx
            ],
            [
                self.tanimoto_similarities[(input_smile, output_smiles[eo])]
                for eo in idx
            ],
        )  # pylint: disable=E1102

    def __len__(self):
        return len(self.source_to_targets)

    @staticmethod
    def collate_fn(encoded_pairs) -> Mol2MolBatchDTO:
        """Turns a list of encoded pairs (input, target) of sequences and turns them into two batches.

        :param: A list of pairs of encoded sequences.
        :return: A tuple with two tensors, one for the input and one for the targets in the same order as given.
        """

        sources, targets, tanimoto = [], [], []
        for s, t, ts in encoded_pairs:
            sources += s
            targets += t
            tanimoto += ts

        # encoded_inputs, encoded_targets = list(zip(*encoded_pairs))
        collated_arr_source, src_mask = _mask_batch(sources)
        collated_arr_target, trg_mask = _mask_batch(targets)
        tanimoto = torch.tensor(tanimoto, device=DEVICE)

        # TODO: refactor the logic below
        trg_mask = trg_mask & Variable(
            subsequent_mask(collated_arr_target.size(-1)).type_as(trg_mask)
        )
        trg_mask = trg_mask[:, :-1, :-1]  # save start token, skip end token

        dto = Mol2MolBatchDTO(
            collated_arr_source,
            src_mask,
            collated_arr_target,
            trg_mask,
            tanimoto,
        )
        return dto


def _mask_batch(encoded_seqs: List) -> Tuple[Tensor, Tensor]:
    """Pads a batch.

    :param encoded_seqs: A list of encoded sequences.
    :return: A tensor with the sequences correctly padded and masked
    """

    # maximum length of input sequences
    max_length_source = max([seq.size(0) for seq in encoded_seqs])

    # padded source sequences with zeroes
    collated_arr_seq = torch.zeros(
        len(encoded_seqs), max_length_source, dtype=torch.long, device=DEVICE
    )
    seq_mask = torch.zeros(
        len(encoded_seqs), 1, max_length_source, dtype=torch.bool, device=DEVICE
    )

    for i, seq in enumerate(encoded_seqs):
        collated_arr_seq[i, : len(seq)] = seq
        seq_mask[i, 0, : len(seq)] = True

    # mask of source seqs
    # seq_mask = (collated_arr_seq != 0).unsqueeze(-2)

    return collated_arr_seq, seq_mask
