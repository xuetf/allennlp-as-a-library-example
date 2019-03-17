# pylint: disable=invalid-name,protected-access
from flaky import flaky
import pytest

from allennlp.common.testing import ModelTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.models import Model

from libraries import Conll2000DatasetReader
from libraries import CrfTagger

# python3 -m unittest tests/pos_tagger/models/crf_tagger_test.py

class CrfTaggerTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model('tests/pos_tagger/fixtures/experiment_conll2000.json',
                          'tests/pos_tagger/fixtures/conll2000.txt')

    def test_simple_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)


    @flaky
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        tags = output_dict['tags']
        assert len(tags) == 2
        assert len(tags[0]) == 37
        assert len(tags[1]) == 27
        for example_tags in tags:
            for tag_id in example_tags:
                tag = self.model.vocab.get_token_from_index(tag_id, namespace="labels")
                assert tag in {'I-NP','B-NP','I-VP','B-PP','O','B-VP','B-SBAR','B-ADJP'}

    def test_mismatching_dimensions_throws_configuration_error(self):
        params = Params.from_file(self.param_file)
        # Make the encoder wrong - it should be 2 to match
        # the embedding dimension from the text_field_embedder.
        params["model"]["encoder"]["input_size"] = 10
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.pop("model"))
