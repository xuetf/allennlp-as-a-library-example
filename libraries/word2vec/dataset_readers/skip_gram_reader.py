from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields import LabelField
from allennlp.data.instance import Instance
import random
import math
from overrides import overrides
import logging


@DatasetReader.register("skip_gram_text8")
class SkipGramReader(DatasetReader):
    def __init__(self, window_size=5, lazy=False, vocab: Vocabulary=None):
        """A DatasetReader for reading a plain text corpus and producing instances
        for the SkipGram model.
        When vocab is not None, this runs sub-sampling of frequent words as described
        in (Mikolov et al. 2013).
        """
        super().__init__(lazy=lazy)
        self.window_size = window_size
        self.reject_probs = None
        self.target_token_namespace = 'token_target'
        self.context_token_namespace = 'token_context'

        if vocab:
            self.reject_probs = {}
            threshold = 1.e-3
            token_counts = vocab._retained_counter[self.target_token_namespace]  # HACK
            total_counts = sum(token_counts.values())
            for _, token in vocab.get_index_to_token_vocabulary(self.target_token_namespace).items():
                counts = token_counts[token]
                if counts > 0:
                    normalized_counts = counts / total_counts
                    reject_prob = 1. - math.sqrt(threshold / normalized_counts)
                    reject_prob = max(0., reject_prob)
                else:
                    reject_prob = 0.
                self.reject_probs[token] = reject_prob

    def _subsample_tokens(self, tokens):
        """Given a list of tokens, runs sub-sampling.
        Returns a new list of tokens where rejected tokens are replaced by Nones.
        """
        new_tokens = []
        for token in tokens:
            reject_prob = self.reject_probs.get(token, 0.)
            if random.random() <= reject_prob:
                new_tokens.append(None)
            else:
                new_tokens.append(token)

        return new_tokens

    @overrides
    def _read(self, file_path: str):
        with open(file_path, "r") as text_file:
            # only one line
            for line in text_file:
                tokens = line.strip().split(' ')
                tokens = tokens[:10000]  # 1700w+ words, just for the speed of running

                if self.reject_probs:
                    tokens = self._subsample_tokens(tokens)
                    print(tokens[:200])  # for debugging

                for i, token in enumerate(tokens):
                    if token is None:
                        continue

                    for j in range(i - self.window_size, i + self.window_size + 1):
                        if j < 0 or i == j or j > len(tokens) - 1:
                            continue

                        if tokens[j] is None:
                            continue

                        yield self.text_to_instance(token, tokens[j])

    @overrides
    def text_to_instance(self, target, context) -> Instance:
        token_target = LabelField(target, label_namespace=self.target_token_namespace)
        token_context = LabelField(context, label_namespace=self.context_token_namespace)
        return Instance({self.target_token_namespace: token_target,
                         self.context_token_namespace: token_context})

