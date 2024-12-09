from pepinvent.scoring_function.component_names_enum import ComponentNamesEnum
from pepinvent.scoring_function.scoring_components.general.matching_substructure import MatchingSubstructure
from pepinvent.scoring_function.scoring_components.physchem.custom_alerts import CustomAlerts
from pepinvent.scoring_function.scoring_components.physchem.lipophilicity import Lipophilicity
from pepinvent.scoring_function.scoring_components.physchem.max_ring_size import MaxRingSize
from pepinvent.scoring_function.scoring_components.physchem.mol_weight import MolecularWeight
from pepinvent.scoring_function.scoring_components.predictive_model import PredictiveModel
from pepinvent.scoring_function.scoring_components.scoring_component_parameters import ScoringComponentParameters


class ScoringComponentFactory:
    def __init__(self):
        self._registry = {
            ComponentNamesEnum.MaxRingSize: MaxRingSize,
            ComponentNamesEnum.MatchingSubstructure: MatchingSubstructure,
            ComponentNamesEnum.PredictiveModel: PredictiveModel,
            ComponentNamesEnum.Lipophilicity: Lipophilicity,
            ComponentNamesEnum.CustomAlerts: CustomAlerts,
            ComponentNamesEnum.MolecularWeight: MolecularWeight,
        }

    def create_scoring_component(self, component_parameters: ScoringComponentParameters):
        scoring_component = self._registry.get(component_parameters.name)
        instance = scoring_component(component_parameters)
        return instance
