from allennlp.models.model import Model
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.data.vocabulary import Vocabulary
import torch
import numpy as np
from torch.nn import functional as F
import math


@Model.register("skip_gram_negative_sampling")
class SkipGramNegativeSamplingModel(Model):
    def __init__(self, vocab: Vocabulary,
                 embedding_target: TokenEmbedder,
                 embedding_context: TokenEmbedder,
                 neg_samples=10, cuda_device=-1):
        super().__init__(vocab)
        self.embedding_target = embedding_target
        self.embedding_context = embedding_context
        self.neg_samples = neg_samples
        self.cuda_device = cuda_device

        # Pre-compute probability for negative sampling
        if vocab is not None and 'token_target' in vocab._retained_counter:
            token_to_probs = {}
            token_counts = vocab._retained_counter['token_target']  # HACK
            total_counts = sum(token_counts.values())
            total_probs = 0.
            for token, counts in token_counts.items():
                unigram_freq = counts / total_counts
                unigram_freq = math.pow(unigram_freq, 0.75)
                token_to_probs[token] = unigram_freq
                total_probs += unigram_freq

            self.neg_sample_probs = np.ndarray((vocab.get_vocab_size('token_target'),))
            for token_id, token in vocab.get_index_to_token_vocabulary('token_target').items():
                self.neg_sample_probs[token_id] = token_to_probs.get(token, 0) / total_probs

        else:
            print('You need to construct vocab from instances to record the token count statistics')

    def forward(self, token_target, token_context):
        batch_size = token_context.shape[0]

        # Calculate loss for positive examples
        embedded_target = self.embedding_target(token_target)
        embedded_context = self.embedding_context(token_context)
        inner_positive = torch.mul(embedded_target, embedded_context).sum(dim=1)
        log_prob = F.logsigmoid(inner_positive)

        # Generate negative examples, not good, write in iterator is proper
        negative_context = np.random.choice(a=self.vocab.get_vocab_size('token_target'),
                                        size=batch_size * self.neg_samples,
                                        p=self.neg_sample_probs)
        negative_context = torch.LongTensor(negative_context).view(batch_size, self.neg_samples)
        if self.cuda_device > -1:
            negative_context = negative_context.to(self.cuda_device)

        # Subtract loss for negative examples
        embedded_negative_context = self.embedding_context(negative_context)
        inner_negative = torch.bmm(embedded_negative_context, embedded_target.unsqueeze(2)).squeeze()
        log_prob += F.logsigmoid(-1. * inner_negative).sum(dim=1)

        return {'loss': -log_prob.sum() / batch_size}

