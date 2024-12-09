from typing import List, Dict, Union

import torch.utils.data as tud
import pandas as pd

from pepinvent.reinforcement.chemistry import Chemistry
from reinvent_models.model_factory.dto.sampled_sequence_dto import SampledSequencesDTO

from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_models.model_factory.mol2mol_adapter import Mol2MolAdapter
from reinvent_models.mol2mol.dataset.dataset import Dataset
from pepinvent.sampling.sampling_config import SamplingConfig


class Sampling:
    def __init__(self, agent: Union[GenerativeModelBase, Mol2MolAdapter], config: SamplingConfig, ):
        self._agent = agent
        self._config = config
        self._chemistry = Chemistry()

    def _sample(self, sequences: List[str]) -> Dict[str, List[SampledSequencesDTO]]:
        sequence_dtos = {}
        for sequence in sequences:
            input_sequences = self._config.num_samples * [sequence]

            dataset = Dataset(input_sequences, vocabulary=self._agent.vocabulary, tokenizer=self._agent.tokenizer)
            data_loader = tud.DataLoader(
                dataset, self._config.batch_size, shuffle=False, collate_fn=Dataset.collate_fn
            )

            for batch in data_loader:
                src, src_mask = batch
                dtos = self._agent.sample(src, src_mask)
                for dto in dtos:
                    try:
                        sequence_dtos[dto.input].append(dto)
                    except:
                        sequence_dtos[dto.input] = [dto]
        return sequence_dtos

    def execute(self):
        masked_peptides = self.load_data()
        dtos = self._sample(masked_peptides)
        report = self._create_report(dtos)
        self._save_reports(report)

    def load_data(self) -> List[str]:
        test_data = pd.read_csv(self._config.input_sequences_path)
        masked_peptides = list(test_data['Source_Mol'])
        return masked_peptides

    def _create_report(self, results: Dict[str, List[SampledSequencesDTO]]) -> pd.DataFrame:
        summary = []
        dataframe = pd.DataFrame(columns=['Input', 'Output', 'NLLs'])
        for key in results.keys():
            outputs = [dto.output for dto in results[key]]
            nlls = [dto.nll for dto in results[key]]
            # peptides = [self._chemistry.fill_source_peptide(key, output) for output in outputs]
            new_row = {'Input': key, 'Output': outputs, 'NLLs': nlls}
            #  = {'Input': key, 'Peptide': ",".join(peptides), 'Output': ",".join(outputs), 'NLLs': nlls}
            # dataframe["Input"].append(key)
            # dataframe["Peptide"].append(",".join(peptides))
            # dataframe["Output"].append(",".join(outputs))
            # dataframe["Nlls"].append(",".join(nlls))
            dataframe = dataframe.append(new_row, ignore_index=True)

        column_labels = [f'Generated_smi_{x}' for x in range(1, len(results[key]) + 1)]
        split = pd.DataFrame(dataframe['Output'].to_list(), columns=column_labels)
        dataframe = pd.concat([dataframe, split], axis=1)
        dataframe = dataframe.drop('Output', axis=1)

        # for peptide, output, nll in zip(peptides, outputs, nlls):
            #     report_line = f"{key},{peptide},{output},{nll}\n"
            #     summary.append(report_line)
        return dataframe

    def _save_reports(self, output: pd.DataFrame):
        output.to_csv(self._config.results_output)
        # with open(self._config.results_output, "w") as o:
        #     o.writelines(output)
