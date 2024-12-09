from concurrent import futures

from rdkit import RDLogger
from tqdm import tqdm
import numpy as np
import pandas as pd
from reinvent_chemistry.conversions import Conversions
from reinvent_chemistry.similarity import Similarity

from .abstract_generator import PairGenerator

RDLogger.DisableLog("rdApp.info")
RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


class TanimotoPairGenerator(PairGenerator):
    """Generator of molecule pairs according to Tanimoto similarity"""

    def __init__(
        self,
        lower_threshold: float,
        upper_threshold: float = 1.0,
        add_same: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """__init__.

        :param lower_threshold: keeps all the pairs such that tanimoto(s,t) >= lower_threshold
        :type lower_threshold: float
        :param upper_threshold: keeps all the pairs such that tanimoto(s,t) <= upper_threshold
        :type upper_threshold: float
        :param add_same: whether to inlcude the pairs (s,s) or not
        :type add_same: bool
        :rtype: None
        """
        super().__init__(*args, **kwargs)
        assert 0.0 <= lower_threshold <= 1.0
        assert 0.0 <= upper_threshold <= 1.0
        assert lower_threshold < upper_threshold

        self.lth = lower_threshold
        self.uth = upper_threshold
        self.add_same = add_same

    def build_pairs(self, smiles: pd.DataFrame, *, processes: int = 8) -> pd.DataFrame:
        """build_pairs.

        :param smiles: DataFrame containing smiles
        :type smiles: pd.DataFrame
        :param processes: number of process for parallelizing the construction of pairs
        :type processes: int
        :rtype: pd.DataFrame
        """
        assert len(smiles) > 0
        lth = self.lth
        uth = self.uth

        smiles = smiles[smiles.columns[0]].values
        data = self._standardize_smiles(smiles)

        conversions = Conversions()
        fsmiles, fmolecules = [], []

        pbar = tqdm(data, ascii=True)
        pbar.set_description("Smiles to Fingerprints")
        for smi in pbar:
            mol = conversions.smiles_to_fingerprints([smi])
            if len(mol):
                fmolecules.append(mol[0])
                fsmiles.append(smi)
        fsmiles = np.array(fsmiles)
        del data

        data_pool = []
        mol_chunks = np.array_split(fmolecules, processes)
        smile_chunks = np.array_split(fsmiles, processes)
        for pid, (mchunk, schunk) in enumerate(zip(mol_chunks, smile_chunks)):
            data_pool.append(
                {
                    "molecules": mchunk,
                    "smiles": schunk,
                    "mol_db": fmolecules,
                    "smi_db": fsmiles,
                    "lth": lth,
                    "uth": uth,
                    "pid": pid,
                }
            )
        pool = futures.ProcessPoolExecutor(max_workers=processes)
        res = list(pool.map(self._build_pairs, data_pool))
        res = sorted(res, key=lambda x: x["pid"])

        pd_cols = ["Source_Mol", "Target_Mol", "Tanimoto"]
        pd_data = []
        for r in res:
            pd_data = pd_data + r["table"]
        df = pd.DataFrame(pd_data, columns=pd_cols)
        df = df.drop_duplicates(subset=["Source_Mol", "Target_Mol"])
        df = self.filter(df)
        return df

    def _build_pairs(self, args):
        mol_db = args["mol_db"]
        smi_db = args["smi_db"]
        mols = args["molecules"]
        smiles = args["smiles"]
        lth = args["lth"]
        uth = args["uth"]
        pid = args["pid"]
        similarity = Similarity()

        table = []

        for i, mol in tqdm(enumerate(mols), total=len(mols), ascii=True):
            ts = similarity.calculate_tanimoto_batch(mol, mol_db)
            idx = (ts >= lth) & (ts <= uth)
            for smi, t in zip(smi_db[idx], ts[idx]):
                table.append([smiles[i], smi, t])
                table.append([smi, smiles[i], t])
                if self.add_same:
                    table.append([smi, smi, 1.0])
                    table.append([smiles[i], smiles[i], 1.0])
        return {"table": table, "pid": pid}
