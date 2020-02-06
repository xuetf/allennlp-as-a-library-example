from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields import LabelField
from allennlp.data.instance import Instance
import random
import math
from overrides import overrides
import logging


@DatasetReader.register("simple_skip_gram_text8")
class SimpleSkipGramReader(DatasetReader):
    def __init__(self, window_size=5, lazy=False):
        """A DatasetReader for reading a plain text corpus and producing instances
        for the SkipGram model.
        When vocab is not None, this runs sub-sampling of frequent words as described
        in (Mikolov et al. 2013).
        """
        super().__init__(lazy=lazy)
        self.window_size = window_size
        self.target_token_namespace = 'token_target'
        self.context_token_namespace = 'token_context'

    @overrides
    def _read(self, file_path: str):
        with open(file_path, "r") as text_file:
            # only one line
            for line in text_file:
                tokens = line.strip().split(' ')
                tokens = tokens[:100000]  # 1700w+ words, just for the speed of running

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

