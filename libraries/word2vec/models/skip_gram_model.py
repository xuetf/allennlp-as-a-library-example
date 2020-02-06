from allennlp.models.model import Model
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.data.vocabulary import Vocabulary
import torch
from torch.nn import functional as F


@Model.register("skip_gram")
class SkipGramModel(Model):
    def __init__(self, vocab: Vocabulary,
                 embedding_target: TokenEmbedder, cuda_device=-1):
        super().__init__(vocab)
        self.embedding_target = embedding_target
        self.linear = torch.nn.Linear(
            in_features=embedding_target.output_dim,
            out_features=vocab.get_vocab_size('token_context'),
            bias=False)
        if cuda_device > -1:
            self.linear = self.linear.to(cuda_device)

    def forward(self, token_target, token_context):
        embedded_in = self.embedding_in(token_target)
        logits = self.linear(embedded_in)
        loss = F.cross_entropy(logits, token_context)

        return {'loss': loss}

