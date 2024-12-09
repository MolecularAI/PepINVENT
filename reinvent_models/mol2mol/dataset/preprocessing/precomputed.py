from rdkit import RDLogger
import pandas as pd

from .abstract_generator import PairGenerator

RDLogger.DisableLog("rdApp.info")
RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


class PrecomputedPairGenerator(PairGenerator):
    """Generator of molecule pairs according to Tanimoto similarity"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def build_pairs(self, smiles: pd.DataFrame, *, processes: int = 8) -> pd.DataFrame:
        """build_pairs.

        :param smiles: a DataFrame containing smiles
        :type smiles: pd.DataFrame
        :param processes: number of process for parallelizing the construction of pairs
        :type processes: int
        :rtype: pd.DataFrame
        """
        assert len(smiles) > 0
        pd_data = smiles[smiles.columns[:2]].drop_duplicates().values
        pd_cols = ["Source_Mol", "Target_Mol"]
        df = pd.DataFrame(pd_data, columns=pd_cols)
        df = self.filter(df)
        return df
