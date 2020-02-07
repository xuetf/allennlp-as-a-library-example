from collections import deque, defaultdict
from typing import Iterable, Deque, Dict
import logging
import random
import math
import numpy as np
from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.fields import MultiLabelField
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset import Batch
from allennlp.common.tqdm import Tqdm

logger = logging.getLogger(__name__)


@DataIterator.register("skip_gram_negative_iterator")
class SkipGramIterator(DataIterator):
    def __init__(self,
                 batch_size: int = 32,
                 neg_samples: int = 10):
        super().__init__(batch_size)
        self.counter = None
        self.neg_sample_probs = None
        self.neg_samples = neg_samples

    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        if self.counter is None:
            self.build_counter(instances)

        # First break the dataset into memory-sized lists:
        for instance_list in self._memory_sized_lists(instances):
            if shuffle:
                random.shuffle(instance_list)
            iterator = iter(instance_list)
            excess: Deque[Instance] = deque()
            # Then break each memory-sized list into batches.
            for batch_instances in lazy_groups_of(iterator, self._batch_size):
                batch_instances = self.modify_batch_instances(batch_instances)
                for possibly_smaller_batches in self._ensure_batch_is_sufficiently_small(batch_instances, excess):
                    batch = Batch(possibly_smaller_batches)
                    yield batch
            if excess:
                yield Batch(excess)

    def build_counter(self, instances: Iterable[Instance]):
        namespace_token_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for instance in Tqdm.tqdm(instances):
            instance.count_vocab_items(namespace_token_counts)
        self.counter = namespace_token_counts

    def build_prob(self):
        if self.vocab:
            token_namespace = 'token_target'
            token_to_probs = {}
            token_counts = self.counter[token_namespace]  # HACK
            total_counts = sum(token_counts.values())
            total_probs = 0.
            for token, counts in token_counts.items():
                unigram_freq = 1.0 * counts / total_counts
                unigram_freq = math.pow(unigram_freq, 0.75)
                token_to_probs[token] = unigram_freq
                total_probs += unigram_freq

            self.neg_sample_probs = np.ndarray((self.vocab.get_vocab_size(token_namespace),))
            for token_id, token in self.vocab.get_index_to_token_vocabulary(token_namespace).items():
                self.neg_sample_probs[token_id] = token_to_probs.get(token, 0) / total_probs

    def get_negative_contexts(self, batch_size):
        if self.vocab is None:
            print('vocab cannot be empty')
            return
        token_namespace = 'token_context'
        if self.neg_sample_probs is None:
            self.build_prob()

        negative_index_context = np.random.choice(a=self.vocab.get_vocab_size(token_namespace),
                                            size=(batch_size, self.neg_samples),
                                            p=self.neg_sample_probs)
        negative_token_context_list = []
        for batch_neg_ind in negative_index_context:
            negative_tokens = []
            for neg_ind in batch_neg_ind:
                negative_tokens.append(self.vocab.get_token_from_index(neg_ind, token_namespace))
            negative_token_context_list.append(negative_tokens)

        return negative_token_context_list

    def modify_batch_instances(self, batch_instances):
        batch_instances = list(batch_instances)
        batch_size = len(batch_instances)
        negative_token_contexts = self.get_negative_contexts(batch_size)
        token_namespace = 'token_context'
        for instance, negs in zip(batch_instances, negative_token_contexts):
            instance.add_field('negative_context', MultiLabelField(negs, label_namespace=token_namespace))
        return batch_instances
