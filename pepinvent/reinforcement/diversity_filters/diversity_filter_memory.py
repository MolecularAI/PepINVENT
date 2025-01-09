from typing import List, Dict

import pandas as pd

from reinvent_scoring.scoring.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum

from pepinvent.scoring_function.score_summary import ComponentSummary


class DiversityFilterMemory:

    def __init__(self):
        self._sf_component_name = ScoringFunctionComponentNameEnum()
        df_dict = {"Step": [], "CHUCKLES": [], "SMILES": [], "Scaffold": []}
        self._memory_dataframe = pd.DataFrame(df_dict)

    def update(self, indx: int, score: float, smile: str, sampled_sequences: str, components: List[ComponentSummary], step: int, scaffold: str = ''):
        component_scores = {c.parameters.name: float(c.total_score[indx]) for c in components}
        component_scores = self._include_raw_score(indx, component_scores, components)
        component_scores[self._sf_component_name.TOTAL_SCORE] = float(score)
        if not self.smiles_exists(smile):
            self._add_to_memory_dataframe(step, smile, sampled_sequences, component_scores, scaffold)

    def _add_to_memory_dataframe(self, step: int, smile: str, sampled_sequences: str, component_scores: Dict, scaffold: str = ''):
        data = []
        headers = []
        for name, score in component_scores.items():
            headers.append(name)
            data.append(score)
        headers.append("Step")
        data.append(step)
        headers.append("CHUCKLES")
        data.append(sampled_sequences)
        headers.append("SMILES")
        data.append(smile)
        headers.append("Scaffold")
        data.append(scaffold)
        new_data = pd.DataFrame([data], columns=headers)
        self._memory_dataframe = pd.concat([self._memory_dataframe, new_data], ignore_index=True, sort=False)

    def get_memory(self) -> pd.DataFrame:
        return self._memory_dataframe

    def set_memory(self, memory: pd.DataFrame):
        self._memory_dataframe = memory

    def smiles_exists(self, smiles: str):
        if len(self._memory_dataframe) == 0:
            return False
        return smiles in self._memory_dataframe['SMILES'].values

    def scaffold_instances_count(self, scaffold: str):
        return (self._memory_dataframe["Scaffold"].values == scaffold).sum()

    # def number_of_scaffolds(self):
    #     return len(set(self._memory_dataframe["CHUCKLES"].values))

    def number_of_smiles(self):
        return len(set(self._memory_dataframe["SMILES"].values))

    def _include_raw_score(self, indx: int, component_scores: dict, components: List[ComponentSummary]):
        raw_scores = {f'raw_{c.parameters.name}': float(c.raw_score[indx]) for c in components if
                      c.raw_score is not None}
        all_scores = {**component_scores, **raw_scores}
        return all_scores
