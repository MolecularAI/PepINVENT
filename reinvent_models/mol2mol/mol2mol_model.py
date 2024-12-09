from typing import Union, Any

import torch
from dacite import from_dict
from torch import nn as tnn
from torch.autograd import Variable

from reinvent_models.model_factory.enums.model_mode_enum import ModelModeEnum
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase

from reinvent_models.model_factory.dto.sampled_sequence_dto import SampledSequencesDTO
from reinvent_models.mol2mol.dto.mol2mol_model_parameters_dto import Mol2MolModelParameterDTO
from reinvent_models.mol2mol.enums.sampling_mode_enum import SamplingModesEnum

from reinvent_models.mol2mol.models.encode_decode.model import EncoderDecoder
from reinvent_models.mol2mol.models.module.subsequent_mask import subsequent_mask

from reinvent_models.mol2mol.models.module.search import beamsearch, Node, EOS, MaxLength, LogicalOr
from reinvent_models.mol2mol.models.vocabulary import SMILESTokenizer, Vocabulary


class Mol2MolModel:
    def __init__(self, vocabulary: Vocabulary, network: EncoderDecoder,
                 max_sequence_length: int = 128, no_cuda: bool = False, mode: str = ModelModeEnum().TRAINING):
        self.vocabulary = vocabulary
        self.tokenizer = SMILESTokenizer()
        self.network = network
        self._model_modes = ModelModeEnum()
        self._sampling_modes_enum = SamplingModesEnum()
        self.max_sequence_length = max_sequence_length

        self.set_mode(mode)
        if torch.cuda.is_available() and not no_cuda:
            self.network.cuda()

        self.device = next(self.network.parameters()).device
        self._nll_loss = tnn.NLLLoss(reduction="none", ignore_index=0)
        
        self.beam_size = 64

        self.temperature = 1.0
 
    def set_beam_size(self, beam_size):
        self.beam_size = beam_size

    def set_temperature(self, temperature: float=1.0):
        self.temperature = temperature

    def set_mode(self, mode: str):
        if mode == self._model_modes.TRAINING:
            self.network.train()
        elif mode == self._model_modes.INFERENCE:
            self.network.eval()
        else:
            raise ValueError(f"Invalid model mode '{mode}")

    @classmethod
    def load_from_file(cls, path_to_file, mode: str = ModelModeEnum().TRAINING) -> Union[Any, GenerativeModelBase] :
        """
        Loads a model from a single file
        :param path_to_file: Path to the saved model
        :param mode: Mode in which the model should be initialized
        :return: An instance of the network
        """
        data = from_dict(Mol2MolModelParameterDTO, torch.load(path_to_file))
        network = EncoderDecoder(**vars(data.network_parameter))
        network.load_state_dict(data.network_state)
        model = cls(vocabulary=data.vocabulary, network=network,
                    max_sequence_length=data.max_sequence_length, mode=mode)
        return model

    def save_to_file(self, path_to_file):
        """
        Saves the model to a file.
        :param path_to_file: Path to the file which the model will be saved to.
        """
        data = Mol2MolModelParameterDTO(vocabulary=self.vocabulary, max_sequence_length=self.max_sequence_length,
                                          network_parameter=self.network.get_params(),
                                          network_state=self.network.state_dict())
        torch.save(data.__dict__, path_to_file)

    def likelihood(self, src, src_mask, trg, trg_mask):
        """
        Retrieves the likelihood of molecules.
        :param src: (batch, seq) A batch of padded input sequences.
        :param src_mask: (batch, seq, seq) Mask of the input sequences.
        :param trg: (batch, seq) A batch of output sequences; with start token, without end token.
        :param trg_mask: Mask of the input sequences.
        :return:  (batch) Log likelihood for each output sequence in the batch.
        """
        trg_y = trg[:, 1:] # skip start token but keep end token
        trg = trg[:, :-1] # save start token, skip end token
        out = self.network.forward(src, trg, src_mask, trg_mask)
        log_prob = self.network.generator(out, self.temperature).transpose(1, 2) #(batch, voc, seq_len)
        nll = self._nll_loss(log_prob, trg_y).sum(dim=1)

        return nll

    @torch.no_grad()
    def sample(self, src, src_mask, decode_type):
        """
        Sample molecules
        :param src: (batch, seq) A batch of padded input sequences.
        :param src_mask: (batch, seq, seq) Mask of the input sequences.
        :param decode_type: decode type
        """
        beam_size = self.beam_size
        if decode_type == self._sampling_modes_enum.BEAMSEARCH:
            vocabulary = self.vocabulary
            tokenizer = self.tokenizer
            vocabulary.pad_token = 0  # 0 is padding
            vocabulary.bos_token = 1  # 1 is start symbol
            vocabulary.eos_token = 2  # 2 is end symbol

            stop_criterion = LogicalOr((MaxLength(self.max_sequence_length - 1), EOS()))
            node = Node(self.network, (src, src_mask), vocabulary, self.device,
                        batch_size=len(src),
                        data_device=self.device) # if it explodes use 'cpu' here
            beamsearch(node, beam_size, stop_criterion)
            output_smiles_list = [tokenizer.untokenize(self.vocabulary.decode(seq)) for seq in
                                  node.y.detach().cpu().numpy()]
            input_smiles_list = []
            for seq in src.detach().cpu().numpy():
                s = tokenizer.untokenize(self.vocabulary.decode(seq))
                for _ in range(beam_size):
                    input_smiles_list.append(s)
            nlls = (-node.loglikelihood.detach().cpu().numpy()).ravel()
            result = [SampledSequencesDTO(input, output, nll) for input, output, nll in
                      zip(input_smiles_list, output_smiles_list,
                          nlls.tolist())]
        else:
            batch_size = src.shape[0]
            ys = torch.ones(1).to(self.device)
            ys = ys.repeat(batch_size, 1).view(batch_size, 1).type_as(src.data) # shape [batch_size, 1]
            encoder_outputs = self.network.encode(src, src_mask)
            break_condition = torch.zeros(batch_size, dtype=torch.bool).to(self.device)

            nlls = torch.zeros(batch_size).to(self.device)
            # FIXME: end_token = self.vocabulary.end_token
            end_token = self.vocabulary["$"]
            for i in range(self.max_sequence_length - 1):
                out = self.network.decode(encoder_outputs, src_mask, Variable(ys),
                                          Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
                # (batch, seq, voc) need to exclude the probability of the start token "1"
                log_prob = self.network.generator(out[:, -1], self.temperature)
                prob = torch.exp(log_prob)

                mask_property_token = self.mask_property_tokens(batch_size)
                prob = prob.masked_fill(mask_property_token, 0)

                if decode_type == self._sampling_modes_enum.GREEDY:
                    _, next_word = torch.max(prob, dim=1)
                    # mask numbers after end token as 0
                    next_word = next_word.masked_fill(break_condition.to(self.device), 0)
                    ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)  # [batch_size, i]

                    # This does the same as loss(), logprob = torch.index_select(log_prob, 1, next_word)
                    nlls += self._nll_loss(log_prob, next_word)
                elif decode_type == self._sampling_modes_enum.MULTINOMIAL:
                    next_word = torch.multinomial(prob, 1)
                    # mask numbers after end token as 0
                    break_t = torch.unsqueeze(break_condition, 1).to(self.device)
                    next_word = next_word.masked_fill(break_t, 0)
                    ys = torch.cat([ys, next_word], dim=1)  # [batch_size, i]
                    next_word = torch.reshape(next_word, (next_word.shape[0],))

                    # This does the same as loss(), logprob = torch.index_select(log_prob, 1, next_word)
                    nlls += self._nll_loss(log_prob, next_word)

                # next_word = np.array(next_word.to('cpu').tolist())
                break_condition = (break_condition | (next_word == end_token))
                if all(break_condition):  # end token
                    break

            tokenizer = SMILESTokenizer()
            output_smiles_list = [tokenizer.untokenize(self.vocabulary.decode(seq)) for seq in ys.detach().cpu().numpy()]
            input_smiles_list = [tokenizer.untokenize(self.vocabulary.decode(seq)) for seq in src.detach().cpu().numpy()]
            result = [SampledSequencesDTO(input, output, nll) for input, output, nll in
                      zip(input_smiles_list, output_smiles_list, nlls.detach().cpu().numpy().tolist())]

        return result

    def mask_property_tokens(self, batch_size):
        """
        Prevent model from sampling the property tokens even though it happens very rarely.
        The ChEMBL prior in the paper below was trained with property tokens in the vocabulary.

        He, J., Nittinger, E., Tyrchan, C., Czechtizky, W., Patronov, A., Bjerrum, E. J., & Engkvist, O. (2022).
        Transformer-based molecular optimization beyond matched molecular pairs. Journal of cheminformatics, 14(1), 1-14.
        """
        property_tokens = ['LogD_(3.9, 4.1]', 'LogD_(2.5, 2.7]', 'LogD_(5.9, 6.1]', 'LogD_(-6.1, -5.9]',
                           'LogD_(3.3, 3.5]', 'LogD_(-2.1, -1.9]', 'LogD_(4.7, 4.9]', 'LogD_(-4.5, -4.3]',
                           'LogD_(0.7, 0.9]', 'LogD_(-0.7, -0.5]', 'LogD_(-4.7, -4.5]', 'LogD_(-5.1, -4.9]',
                           'LogD_(-6.5, -6.3]', 'LogD_(3.5, 3.7]', 'Solubility_no_change', 'LogD_(-3.7, -3.5]',
                           'LogD_(-1.9, -1.7]', 'LogD_(-1.5, -1.3]', 'LogD_(-0.3, -0.1]', 'LogD_(6.7, 6.9]',
                           'LogD_(-1.3, -1.1]', 'LogD_(4.3, 4.5]', 'Clint_no_change', 'LogD_(0.3, 0.5]',
                           'LogD_(-5.3, -5.1]', 'LogD_(5.7, 5.9]', 'LogD_(-0.9, -0.7]', 'LogD_(5.3, 5.5]',
                           'LogD_(6.9, inf]', 'LogD_(-3.1, -2.9]', 'LogD_(-3.9, -3.7]', 'LogD_(5.5, 5.7]',
                           'Clint_low->high', 'LogD_(2.3, 2.5]', 'LogD_(2.9, 3.1]', 'LogD_(6.5, 6.7]',
                           'LogD_(-2.7, -2.5]', 'LogD_(-5.5, -5.3]', 'LogD_(1.9, 2.1]', 'LogD_(-3.5, -3.3]',
                           'LogD_(-5.9, -5.7]', 'LogD_(-6.3, -6.1]', 'LogD_(-4.9, -4.7]', 'LogD_(-3.3, -3.1]',
                           'Solubility_high->low', 'LogD_(-2.3, -2.1]', 'LogD_(5.1, 5.3]', 'LogD_(-0.1, 0.1]',
                           'LogD_(3.1, 3.3]', 'LogD_(-2.9, -2.7]', 'LogD_(1.1, 1.3]', 'LogD_(-2.5, -2.3]',
                           'Clint_high->low', 'LogD_(-1.1, -0.9]', 'LogD_(4.5, 4.7]', 'LogD_(-inf, -6.9]',
                           'LogD_(6.3, 6.5]', 'LogD_(-6.9, -6.7]', 'LogD_(3.7, 3.9]', 'LogD_(-4.1, -3.9]',
                           'LogD_(1.7, 1.9]', 'LogD_(2.7, 2.9]', 'Solubility_low->high', 'LogD_(4.9, 5.1]',
                           'LogD_(4.1, 4.3]', 'LogD_(-6.7, -6.5]', 'LogD_(-1.7, -1.5]', 'LogD_(0.1, 0.3]',
                           'LogD_(-4.3, -4.1]', 'LogD_(2.1, 2.3]', 'LogD_(-0.5, -0.3]', 'LogD_(0.9, 1.1]',
                           'LogD_(6.1, 6.3]', 'LogD_(0.5, 0.7]', 'LogD_(-5.7, -5.5]', 'LogD_(1.3, 1.5]',
                           'LogD_(1.5, 1.7]']
        mask_property_token = torch.zeros(batch_size, len(self.vocabulary), dtype=torch.bool).to(self.device)
        for p_token in property_tokens:
            if p_token in self.vocabulary:
                i = self.vocabulary[p_token]
                mask_property_token[:, i] = True

        return mask_property_token

    def get_network_parameters(self):
        return self.network.parameters()
