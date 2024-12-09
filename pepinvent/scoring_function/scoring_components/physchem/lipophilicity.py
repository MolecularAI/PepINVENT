from pepinvent.scoring_function.scoring_components.physchem.base_physchem_component import BasePhysChemComponent
from pepinvent.scoring_function.scoring_components.scoring_component_parameters import ScoringComponentParameters


class Lipophilicity(BasePhysChemComponent):
    def __init__(self, parameters: ScoringComponentParameters):
        super().__init__(parameters)

    def _calculate_phys_chem_property(self, mol):
        return self.phys_chem_descriptors.slog_p(mol)
