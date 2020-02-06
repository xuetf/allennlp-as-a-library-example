from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.predictors.predictor import Predictor
from torch.nn import CosineSimilarity
from scipy.stats import spearmanr

def read_simlex999():
    simlex999 = []
    with open('data/word2vec/SimLex-999/SimLex-999.txt') as f:
        next(f)
        for line in f:
            fields = line.strip().split('\t')
            word1, word2, _, sim = fields[:4]
            sim = float(sim)
            simlex999.append((word1, word2, sim))

    return simlex999


@Predictor.register('simplex_predictor')
class SimlexPredictor(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single set of tags for it.  In particular, it can be used with
    the :class:`~allennlp.models.crf_tagger.CrfTagger` model
    and also
    the :class:`~allennlp.models.simple_tagger.SimpleTagger` model.
    """

    @overrides
    def _json_to_instance(self, json_dict: JsonDict):
        """
        Expects JSON that looks like ``{"word": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        return json_dict

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance_dict = self._json_to_instance(inputs)
        instance_dict['Simlex999'] = self.evaluate_embeddings()
        return instance_dict

    def evaluate_embeddings(self):
        vocab = self._model.vocab
        embedding = self._model.embedding_target

        cosine = CosineSimilarity(dim=0)

        simlex999 = read_simlex999()
        sims_pred = []
        oov_count = 0
        for word1, word2, sim in simlex999:
            word1_id = vocab.get_token_index(word1, 'token_target')
            if word1_id == 1:
                sims_pred.append(0.)
                oov_count += 1
                continue
            word2_id = vocab.get_token_index(word2, 'token_target')
            if word2_id == 1:
                sims_pred.append(0.)
                oov_count += 1
                continue

            sim_pred = cosine(embedding.weight[word1_id],
                              embedding.weight[word2_id]).item()
            sims_pred.append(sim_pred)

        assert len(sims_pred) == len(simlex999)
        print('# of OOV words: {} / {}'.format(oov_count, len(simlex999)))

        return spearmanr(sims_pred, [sim for _, _, sim in simlex999])


