from abc import ABC, abstractmethod
from collections import defaultdict

from rdkit import RDLogger
from tqdm import tqdm
import numpy as np
import os
import pandas as pd

from reinvent_chemistry.conversions import Conversions

RDLogger.DisableLog("rdApp.info")
RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


class PairGenerator(ABC):
    def __init__(
        self, min_cardinality: int, max_cardinality: int, *args, **kwargs
    ) -> None:
        """__init__.

        :param min_cardinality: minimum number of targets for each source
        :type min_cardinality: int
        :param max_cardinality: maximum number of targets for each source
        :type max_cardinality: int
        """
        assert min_cardinality <= max_cardinality
        self.min_cardinality = min_cardinality
        self.max_cardinality = max_cardinality

    @abstractmethod
    def build_pairs(self, smiles: list, *, processes: int):
        """Abstract method for building pairs (source,target)


        :param smiles: a list of smiles
        :type smiles: list
        :param processes: number of process for parallelizing the construction of pairs
        :type processes: int
        """
        pass

    def filter(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Keeps all the pairs such that for each source s, min_cardinality <= | { (s, t_i) } | <= max_cardinality.

        :param pairs: DataFrame containing the pairs. It must contain columns "Source_Mol" and "Target_Mol"
        :type pairs: pd.DataFrame
        :rtype: pd.DataFrame
        """

        assert "Source_Mol" in pairs.columns
        assert "Target_Mol" in pairs.columns

        locations = defaultdict(list)
        for i, smi in enumerate(pairs["Source_Mol"]):
            locations[smi].append(i)

        good_locations = []
        for k in locations:
            if (len(locations[k]) >= self.min_cardinality) and (
                len(locations[k]) <= self.max_cardinality
            ):
                good_locations += locations[k]
        good_locations = np.array(good_locations)
        return pairs.iloc[good_locations].reset_index(drop=True)

    def _standardize_smiles(self, smiles):
        conversions = Conversions()
        std_smiles = set()
        pbar = tqdm(smiles)
        pbar.set_description("Standardizing smiles")
        for smi in pbar:
            std_smi = conversions.convert_to_standardized_smiles(smi)
            if (std_smi is not None) and (len(std_smi) > 0):
                std_smiles.add(std_smi)
        return np.array(list(std_smiles))

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__class__.__name__
