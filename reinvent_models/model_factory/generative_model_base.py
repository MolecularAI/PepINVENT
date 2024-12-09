from abc import ABC, abstractmethod


class GenerativeModelBase(ABC):

    @abstractmethod
    def save_to_file(self, path_to_file: str):
        raise NotImplementedError("save_to_file method is not implemented")

    @abstractmethod
    def likelihood(self, *args, **kwargs):
        raise NotImplementedError("likelihood method is not implemented")

    @abstractmethod
    def likelihood_smiles(self, *args, **kwargs):
        raise NotImplementedError("likelihood_smiles method is not implemented")

    @abstractmethod
    def sample(self, *args, **kwargs):
        raise NotImplementedError("sample method is not implemented")

    @abstractmethod
    def set_mode(self, mode: str):
        raise NotImplementedError("set_mode method is not implemented")

    @abstractmethod
    def get_network_parameters(self):
        raise NotImplementedError("get_network_parameters method is not implemented")

    def get_vocabulary(self):
        # FIXME all models return different data structures
        # for lib and link_invent, model.vocabulary is a combination of two vocabularies and their tokenizers
        # link_invent -> PairedModelVocabulary
        # lib_invent -> DecoratorVocabulary
        # reinvent_core -> Vocabulary
        raise NotImplementedError("get_vocabulary method is not implemented")
