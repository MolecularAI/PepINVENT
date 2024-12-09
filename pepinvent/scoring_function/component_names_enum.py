from dataclasses import dataclass


@dataclass(frozen=True)
class ComponentNames:
    MaxRingSize: str = "maximum_ring_size"
    MolecularWeight: str = "molecular_weight"
    MatchingSubstructure: str = "substructure_match"
    PredictiveModel: str = 'predictive_model'
    Lipophilicity: str = 'lipophilicity'
    CustomAlerts: str = 'custom_alerts'

ComponentNamesEnum = ComponentNames()
