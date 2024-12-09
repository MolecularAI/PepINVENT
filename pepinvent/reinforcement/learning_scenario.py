import time
from typing import List, Union, Tuple

import numpy as np
import torch
import torch.utils.data as tud
from reinvent_chemistry import Conversions

from reinvent_models.model_factory.dto.sampled_sequence_dto import SampledSequencesDTO
from reinvent_models.model_factory.mol2mol_adapter import Mol2MolAdapter

from pepinvent.reinforcement.chemistry import Chemistry
from pepinvent.reinforcement.configuration.reinforcement_learning_configuration import ReinforcementLearningConfiguration
from pepinvent.reinforcement.diversity_filters.diversity_filter import DiversityFilter
from pepinvent.reinforcement.dto.likelihood_dto import LikelihoodDTO
from pepinvent.reinforcement.dto.output_dto import OutputDTO
from pepinvent.reinforcement.dto.scoring_input_dto import ScoringInputDTO
from pepinvent.reinvent_logging.local_reinforcement_logger import LocalReinforcementLogger
from reinvent_models.mol2mol.dataset.dataset import Dataset
from pepinvent.scoring_function.score_summary import FinalSummary
from pepinvent.scoring_function.scoring_function_factory import ScoringFunctionFactory


class LearningScenario:
    def __init__(self, agent: Mol2MolAdapter, prior: Mol2MolAdapter, config: ReinforcementLearningConfiguration, ):
        self._agent = agent
        self._prior = prior
        self._config = config
        self._logger = LocalReinforcementLogger(self._config.logging)
        self._optimizer = torch.optim.Adam(self._agent.generative_model.get_network_parameters(),
                                           lr=self._config.learning_configuration.learning_rate)
        self._scoring_function = ScoringFunctionFactory(config.scoring_function).create_scoring_function()
        self._diversity_filter = DiversityFilter(config.diversity_filter)
        self._chemistry = Chemistry()
        self._conversions = Conversions()

    def _sample(self) -> List[SampledSequencesDTO]:
        input = self._config.learning_configuration.batch_size * [self._config.input_sequence]

        dataset = Dataset(input, vocabulary=self._agent.vocabulary, tokenizer=self._agent.tokenizer)
        data_loader = tud.DataLoader(
            dataset, self._config.learning_configuration.batch_size, shuffle=False, collate_fn=Dataset.collate_fn
        )
        for batch in data_loader:
            src, src_mask = batch
            sequence_dtos = self._agent.sample(src, src_mask)
            return sequence_dtos

    def _score(self, sampled_sequences_dto: List[SampledSequencesDTO], step: int) -> Tuple[FinalSummary, List[SampledSequencesDTO]]:
        outputs = [dto.output for dto in sampled_sequences_dto]
        peptides = [self._chemistry.fill_source_peptide(self._config.input_sequence, output) for output in outputs]
        scoring_input = ScoringInputDTO(peptides=peptides, peptide_input=self._config.input_sequence,
                                        peptide_outputs=outputs)
        # uniqueness check
        sorted_indices = self._get_indices_of_unique_smiles(scoring_input.peptides)
        scoring_input_unique, sampled_sequences_dto_unique = self._keep_unique(scoring_input, sorted_indices, sampled_sequences_dto)
        print(f" unique molecules:{len(scoring_input_unique.peptides)}\n")

        final_summary = self._scoring_function.calculate_score(scoring_input)
        final_summary.total_score = self._diversity_filter.update_score(final_summary, sampled_sequences_dto_unique, step)
        return final_summary, sampled_sequences_dto_unique

    def _update(self, final_summary: FinalSummary, sampled_sequences_dto: List[SampledSequencesDTO]) -> LikelihoodDTO:
        agent_likelihood = -self._agent.likelihood_smiles(sampled_sequences_dto).likelihood
        prior_likelihood = -self._prior.likelihood_smiles(sampled_sequences_dto).likelihood

        # score = self._diversity_filter.update_score(final_summary, step)
        distance_penalty = self._get_distance_to_prior(prior_likelihood, self._config.learning_configuration.distance_threshold)


        score_tensor = final_summary.total_score * distance_penalty
        score_tensor = torch.from_numpy(score_tensor)
        score_tensor = self.to_tensor(score_tensor)
        augmented_tensor: torch.Tensor = prior_likelihood + self._config.learning_configuration.score_multiplier * score_tensor
        likelihoods = LikelihoodDTO(prior_likelihood=prior_likelihood, agent_likelihood=agent_likelihood,
                                    augmented_likelihood=augmented_tensor)

        loss = torch.pow((augmented_tensor - agent_likelihood), 2)

        loss = loss.mean()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return likelihoods

    def _log(self, start_time, step, final_summary, likelihoods, report: List[OutputDTO]):
        self._logger.timestep_report(start_time, self._config.learning_configuration.number_steps, step, final_summary,
                                     likelihoods.agent_likelihood,
                                     likelihoods.prior_likelihood, likelihoods.augmented_likelihood, report)

    def execute(self):
        start_time = time.time()

        for step in range(self._config.learning_configuration.number_steps):
            dtos = self._sample()
            final_summary, dtos = self._score(dtos, step)
            likelihoods = self._update(final_summary, dtos)

            report = self._create_report(dtos, final_summary)

            self._log(start_time, step, final_summary, likelihoods, report)

        self._logger.save_final_state(self._diversity_filter.get_memory_as_dataframe())

    def _create_report(self, dtos, final_summary) -> List[OutputDTO]:
        outputs = [dto.output for dto in dtos]
        aminoacids = [self._chemistry.get_generated_amino_acids(output) for output in outputs]
        report = [OutputDTO(peptide=smi, amino_acids=aa) for smi, aa in zip(final_summary.scored_smiles, aminoacids)]
        return report

    def to_tensor(self, tensor):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        if torch.cuda.is_available():
            return torch.autograd.Variable(tensor).cuda()
        return torch.autograd.Variable(tensor)

    @torch.no_grad()
    def _get_distance_to_prior(self, prior_likelihood: Union[torch.Tensor, np.ndarray],
                              distance_threshold=-20.) -> np.ndarray:
        """prior_likelihood and distance_threshold have negative values"""
        if type(prior_likelihood) == torch.Tensor:
            ones = torch.ones_like(prior_likelihood, requires_grad=False)
            mask = torch.where(prior_likelihood > distance_threshold, ones, distance_threshold / prior_likelihood)
            mask = mask.cpu().numpy()
        else:
            ones = np.ones_like(prior_likelihood)
            mask = np.where(prior_likelihood > distance_threshold, ones, distance_threshold / prior_likelihood)
        return mask

    def _get_indices_of_unique_smiles(self, smiles: [str]) -> np.array:
        smiles_list = []
        for indx, smile in enumerate(smiles):
            if self._conversions.smile_to_mol(smile) is not None:
                canonical_smiles = self._chemistry.canonicalize_smiles([smile], isomericSmiles=True)
                smiles_list.append(canonical_smiles[0])
            else:
                smiles_list.append(f"{indx}_INVALID")

        _, idxs = np.unique(np.array(smiles_list), return_index=True)
        sorted_indices = np.sort(idxs)
        return sorted_indices

    def _keep_unique(self, scoring_input: ScoringInputDTO, unique_indices, sampled_sequences_dto: List[SampledSequencesDTO]) -> Tuple[ScoringInputDTO, List[SampledSequencesDTO]]:
        scoring_input.peptide_outputs = [scoring_input.peptide_outputs[idx] for idx in unique_indices]
        scoring_input.peptides = [scoring_input.peptides[idx] for idx in unique_indices]
        sampled_sequences_dto = [sampled_sequences_dto[idx] for idx in unique_indices]
        return scoring_input, sampled_sequences_dto
