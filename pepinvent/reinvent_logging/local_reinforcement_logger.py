import os
from typing import List

import numpy as np
import pandas as pd
#from pepinvent import reinvent_logging as lu
import reinvent_chemistry.logging as lu
import torch
from rdkit import Chem
from torch.utils.tensorboard import SummaryWriter

from pepinvent.reinforcement.dto.output_dto import OutputDTO
from pepinvent.reinvent_logging.base_reinforcement_logger import BaseReinforcementLogger
from pepinvent.reinvent_logging.consolde_message import ConsoleMessage
from pepinvent.reinvent_logging.logging_config import LoggingConfig
from pepinvent.scoring_function.score_summary import FinalSummary


class LocalReinforcementLogger(BaseReinforcementLogger):
    def __init__(self, rl_config: LoggingConfig):
        super().__init__(rl_config)
        self._summary_writer = SummaryWriter(log_dir=self._log_config.logging_path)
        self._summary_writer.add_text('Legends',
                                      'The values under each compound are read as: [Agent; Prior; Target; Score]')
        self._rows = 3
        self._columns = 5
        self._sample_size = self._rows * self._columns
        self._console_message_formatter = ConsoleMessage()

    def log_message(self, message: str):
        self._logger.info(message)

    def timestep_report(self, start_time, n_steps, step, score_summary: FinalSummary,
                        agent_likelihood: torch.tensor, prior_likelihood: torch.tensor,
                        augmented_likelihood: torch.tensor, report: List[OutputDTO]):

        message = self._console_message_formatter.create(start_time, n_steps, step, score_summary.scored_smiles, score_summary.total_score.mean(), score_summary,
                                                         score_summary.total_score, agent_likelihood, prior_likelihood,
                                                         augmented_likelihood)
        self._logger.info(message)
        self._tensorboard_report(step, score_summary.scored_smiles, score_summary.total_score, score_summary, agent_likelihood, prior_likelihood,
                                 augmented_likelihood, report)

    def save_final_state(self, output: pd.DataFrame):
        output.to_csv(os.path.join(self._log_config.result_path, 'results.csv'))
        self._summary_writer.close()

    def _tensorboard_report(self, step, smiles, score, score_summary: FinalSummary, agent_likelihood, prior_likelihood,
                            augmented_likelihood, outputs: List[OutputDTO]):
        self._summary_writer.add_scalars("nll/avg", {
            "prior": prior_likelihood.mean(),
            "augmented": augmented_likelihood.mean(),
            "agent": agent_likelihood.mean()
        }, step)
        mean_score = np.mean(score)
        for i, log in enumerate(score_summary.profile):
            self._summary_writer.add_scalar(score_summary.profile[i].name, np.mean(score_summary.profile[i].score),
                                            step)
        self._summary_writer.add_scalar("Valid SMILES", lu.fraction_valid_smiles(smiles), step)

        self._summary_writer.add_scalar("Average score", mean_score, step)
        if step % 10 == 0:
            self._log_out_smiles_sample(smiles, score, step, score_summary, outputs)

    def _log_out_smiles_sample(self, smiles, score, step, score_summary: FinalSummary, outputs: List[OutputDTO]):
        self._visualize_structures(smiles, score, step, score_summary)
        self._visualize_structures_and_aminoacids(smiles, score, step, score_summary, outputs)

    def _visualize_structures(self, smiles, score, step, score_summary: FinalSummary):

        list_of_mols, legends, pattern = self._check_for_invalid_mols_and_create_legends(smiles, score, score_summary)
        try:
            lu.add_mols(self._summary_writer, "Molecules from epoch", list_of_mols[:self._sample_size], self._rows,
                        [x for x in legends], global_step=step, size_per_mol=(320, 320), pattern=pattern)
        except:
            self.log_message(f"Error in RDKit has occurred, skipping report for step {step}.")

    def _visualize_structures_and_aminoacids(self, smiles, score, step, score_summary: FinalSummary, outputs: List[OutputDTO]):
        aminoacids = [output.amino_acids for output in outputs]
        list_len = [len(i) for i in aminoacids]
        aa_per_row = max(list_len)
        list_of_mols, legends, pattern = self._check_for_invalid_mols_and_create_legends(smiles, score, score_summary)
        reported_mols = []
        for indx, mol in enumerate(list_of_mols):
            if mol:
                aa_mols = [Chem.MolFromSmiles(aa) for aa in aminoacids[indx]]
                if len(aa_mols) == aa_per_row:
                    report = [mol] + aa_mols
                else:
                    filler = [None] * (aa_per_row - len(aa_mols))
                    report = [mol] + aa_mols + filler
            else:
                report = [mol] + [None] * aa_per_row
            reported_mols.extend(report)
        try:
            legends = list(np.repeat(legends,aa_per_row+1))
            mol_for_plot = reported_mols[:self._sample_size*(aa_per_row+1)]
            lu.add_mols(self._summary_writer, "Molecules and Amino Acids from epoch", mol_for_plot, aa_per_row + 1,
                        legends, global_step=step, size_per_mol=(700, 700))
        except:
            self.log_message(f"Error in RDKit has occurred, skipping report for step {step}.")

    def _check_for_invalid_mols_and_create_legends(self, smiles, score, score_summary: FinalSummary):
        smiles = lu.padding_with_invalid_smiles(smiles, self._sample_size)
        list_of_mols, legend = lu.check_for_invalid_mols_and_create_legend(smiles, score, self._sample_size)
        smarts_pattern = ''
        pattern = lu.find_matching_pattern_in_smiles(list_of_mols=list_of_mols, smarts_pattern=smarts_pattern)

        return list_of_mols, legend, pattern
