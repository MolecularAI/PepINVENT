import sys

from rdkit import RDLogger

from .abstract_generator import PairGenerator
from .mmp import MmpPairGenerator
from .precomputed import PrecomputedPairGenerator
from .scaffold import ScaffoldPairGenerator
from .tanimoto import TanimotoPairGenerator

RDLogger.DisableLog("rdApp.info")
RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


def get_pair_generator(pair_generator_name: str, *args, **kwargs) -> PairGenerator:
    """Returns a PairGenerator object.

    :param pair_generator_name: name to retrieve the PairGenerator object. Supported generators: {'tanimoto', 'mmp', 'scaffold', 'precomputed'}
    :type pair_generator_name: str
    :param args: variable positional arguments specific to the PairGenerator
    :param kwargs: variable keyword arguments specific to the PairGenerator
    :rtype: PairGenerator
    """
    pgn = pair_generator_name.strip().capitalize()
    pair_generator = getattr(sys.modules[__name__], f"{pgn}PairGenerator")
    return pair_generator(*args, **kwargs)
