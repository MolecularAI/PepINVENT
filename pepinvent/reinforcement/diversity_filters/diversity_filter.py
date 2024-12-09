from pepinvent.reinforcement.diversity_filters.base_diversity_filter import BaseDiversityFilter
from pepinvent.reinforcement.diversity_filters.diversity_filter_parameters import DiversityFilterParameters
from pepinvent.reinforcement.diversity_filters.identical_murcko_scaffold import IdenticalMurckoScaffold
from pepinvent.reinforcement.diversity_filters.no_filter import NoFilter
from pepinvent.reinforcement.diversity_filters.no_filter_with_penalty import NoFilterWithPenalty


class DiversityFilter:

    def __new__(cls, parameters: DiversityFilterParameters) -> BaseDiversityFilter:
        all_filters = dict(
                           NoFilterWithPenalty=NoFilterWithPenalty,
                           NoFilter=NoFilter,
                           IdenticalMurckoScaffold=IdenticalMurckoScaffold
        )
        div_filter = all_filters.get(parameters.name, KeyError(f"Invalid filter name: `{parameters.name}'"))
        return div_filter(parameters)