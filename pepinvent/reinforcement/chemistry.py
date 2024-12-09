from collections import Counter
from typing import List, Tuple, Union

from rdkit import Chem
from reinvent_chemistry.conversions import Conversions


class Chemistry:
    def __init__(self):
        self._conversions = Conversions()

    # separate this into 2
    def fill_source_peptide(self, source: str, target: str) -> str:
        source = source.split('|')
        target_merge = target.split('|')
        indices = [idx for idx, s in enumerate(source) if s == '?']
        mask_count = source.count('?')
        new_target = []
        m = 0
        if len(target_merge) == mask_count:
            for idx, s in enumerate(source):
                if idx in indices:
                    t = target_merge[m]
                    new_target.append(t)
                    m += 1
                else:
                    new_target.append(s)
            new_target = [aa for aa in new_target[:-1]] + [new_target[-1]]
            new_target = ''.join(new_target)
        else:
            new_target = 'none'
        return new_target

    def check_validity(self, peptide_smiles: List[str]) -> Tuple[List[Chem.Mol], List[int]]:
        valid_mols, valid_idxs = self._conversions.smiles_to_mols_and_indices(peptide_smiles)
        return valid_mols, valid_idxs

    def _find_cyclization_value(self, smiles: str) -> List[str]:
        list_smi = [self._find_integers(x) for x in smiles]
        list_smi = [x for x in list_smi if x]
        list_smi = Counter(list_smi)
        list_smi = [k for k, v in list_smi.items() if v % 2 != 0]
        if len(list_smi) == 0:
            return None
        else:
            return list_smi[0]

    def _strip_cyclization(self, smi: str) -> str:
        tmp_smi = smi
        cyc_nums = self._find_cyclization_value(tmp_smi)
        # if cyc_nums != ['']:
        if cyc_nums:
            for cyc in cyc_nums:
                cyc_idx = smi.index(cyc)
                smi = smi[:cyc_idx] + smi[cyc_idx + 1:]
        return smi

    def _complete_amino_acid(self, smi: str) -> str:
        if not smi.endswith('O'):
            new_smi = smi + 'O'
            mol = Chem.MolFromSmiles(new_smi)
            if mol:
                return new_smi
            else:
                return smi
        else:
            return smi

    @staticmethod
    def _find_integers(x: str) -> Union[int, None]:
        try:
            int(x)
            return x
        except:
            return None

    def _clear_amino_acid(self, aa: str) -> Union[str, None]:
        aa = self._strip_cyclization(aa)
        aa = self._complete_amino_acid(aa)
        if not Chem.MolFromSmiles(aa):
            aa = None
        return aa

    def get_generated_amino_acids(self, output: str) -> List[str]:
        targets = output.split('|')
        targets_cleaned = [self._clear_amino_acid(aa) for aa in targets]
        targets_cleaned = [aa for aa in targets_cleaned if aa]
        return targets_cleaned

    def canonicalize_smiles(self, smiles: List[str], isomericSmiles=False, canonical=True) -> List[str]:
        """This method assumes that all molecules are valid."""
        valid_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=isomericSmiles, canonical=canonical)
                        for smi in smiles]
        return valid_smiles