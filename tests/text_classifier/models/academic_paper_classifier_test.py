# pylint: disable=invalid-name,protected-access
from allennlp.common.testing import ModelTestCase

from libraries import AcademicPaperClassifier # important! registered

# python3 -m unittest tests/text_classifier/models/academic_paper_classifier_test.py
class AcademicPaperClassifierTest(ModelTestCase):
    def setUp(self):
        super(AcademicPaperClassifierTest, self).setUp()
        self.set_up_model('tests/text_classifier/fixtures/academic_paper_classifier.json',
                          'tests/text_classifier/fixtures/s2_papers.jsonl')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

