#import unittest

#from reinvent_models.model_factory.enums import ModelModeEnum
#from reinvent_models.model_factory.enums.model_type_enum import ModelTypeEnum
#from reinvent_models.model_factory.mol2mol_adapter import Mol2MolAdapter
#from reinvent_models.mol2mol.models.vocabulary import Vocabulary
#from reinvent_models import Mol2MolModel


#from reinvent_models.model_factory.enums.model_mode_enum import ModelModeEnum
#from reinvent_models.model_factory.enums.model_type_enum import ModelTypeEnum
#from reinvent_models.model_factory.molformer_adapter import MolformerAdapter
#from reinvent_models.mol2mol.models.vocabulary import Vocabulary
#from reinvent_models.mol2mol.mol2mol_model import Mol2MolModel


#class TestCreateModel(unittest.TestCase):

#     def setUp(self):
#         model_path = "/tmp/pycharm_project_830/examples/models/old_generative_model.chpt"
#         model_type = ModelTypeEnum()
#         model_mode = ModelModeEnum()
#         model: Mol2MolAdapter = Mol2MolAdapter(model_path, model_mode.INFERENCE)
#         vocabulary = model.generative_model.vocabulary
 ##        new_vocabulary = Vocabulary({})
#         new_vocabulary._tokens = vocabulary._tokens
#         new_vocabulary._current_id = vocabulary._current_id
         ##### DONE##
#         network = model.generative_model.network

#         max_sequence_length = model.generative_model.max_sequence_length
#         new_model = Mol2MolModel(vocabulary=new_vocabulary, network=network, max_sequence_length=max_sequence_length)
#         new_model.save_to_file("./generative_model.ckpt")



#     def tearDown(self):
#         pass


#     def test_dataset_length(self):
#         pass
         #torch.save(self.data.__dict__, './pickled_20.chpt')
         # torch.save(self.data.__dict__, './pickled_10_tmp_2.chpt')
