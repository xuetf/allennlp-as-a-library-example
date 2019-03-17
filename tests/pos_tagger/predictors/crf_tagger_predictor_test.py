# pylint: disable=no-self-use,invalid-name,unused-import
from unittest import TestCase

from pytest import approx
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from libraries import CrfTagger
# required so that our custom model + predictor + dataset reader
# will be registered by name

# python3 -m unittest tests/pos_tagger/predictors/crf_tagger_predictor_test.py
class CrfTaggerPredictor(TestCase):
    def test_sentences(self):
        sentence = "We investigate various contextual effects on text"
        archive = load_archive('tests/pos_tagger/fixtures/model.tar.gz') # come from model test (when debugging, copy the output model here)
        predictor = Predictor.from_archive(archive, 'sentence-pos-tagger')
        result = predictor.predict(sentence)

        tags = result.get("tags")
        for tag in tags:
            assert tag in {'I-NP','B-NP','I-VP','B-PP','O','B-VP','B-SBAR','B-ADJP'}

        class_probabilities = result.get("logits")
        assert class_probabilities is not None
        assert len(class_probabilities) == len(tags)
        assert len(tags) == len(sentence.split())

        words = result.get("words")
        for i, word in enumerate(words):
            assert word == sentence.split()[i]
