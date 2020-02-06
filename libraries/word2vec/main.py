from collections import Counter

import torch.optim as optim
from allennlp.data.iterators import BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.training.trainer import Trainer
from torch.nn import CosineSimilarity
from scipy.stats import spearmanr

from libraries import SimpleSkipGramReader
from libraries import SkipGramModel, SkipGramNegativeSamplingModel


EMBEDDING_DIM = 256
WIN_SIZE = 5
BATCH_SIZE = 256
CUDA_DEVICE = -1


def write_embeddings(embedding: Embedding, file_path, vocab: Vocabulary):
    with open(file_path, mode='w') as f:
        for index, token in vocab.get_index_to_token_vocabulary('token_target').items():
            values = ['{:.5f}'.format(val) for val in embedding.weight[index]]
            f.write(' '.join([token] + values))
            f.write('\n')


def get_synonyms(token: str, embedding: Model, vocab: Vocabulary, num_synonyms: int = 10):
    """Given a token, return a list of top N most similar words to the token."""
    token_id = vocab.get_token_index(token, 'token_target')
    token_vec = embedding.weight[token_id]
    cosine = CosineSimilarity(dim=0)
    sims = Counter()

    for index, token in vocab.get_index_to_token_vocabulary('token_target').items():
        sim = cosine(token_vec, embedding.weight[index]).item()
        sims[token] = sim

    return sims.most_common(num_synonyms)


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


def evaluate_embeddings(embedding, vocab: Vocabulary):
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


def build_vocab(data_dir):
    reader = SimpleSkipGramReader(window_size=WIN_SIZE)
    text8 = reader.read(data_dir)
    vocab = Vocabulary.from_instances(text8, min_count={'token_target': 5, 'token_context': 5})
    print('num_token_target={}, num_token_context={}'.format(vocab.get_vocab_size('token_target'),
                                                             vocab.get_vocab_size('token_context')))
    return vocab


def main():
    # "http://mattmahoney.net/dc/text8.zip" download first
    data_dir = 'data/word2vec/text8/text8'

    # 1. build vocab from file
    vocab = build_vocab(data_dir)

    # 2. build reader
    reader = SimpleSkipGramReader(window_size=WIN_SIZE)  # or SkipGramReader(vocab=vocab)
    text8 = reader.read(data_dir)

    embedding_in = Embedding(num_embeddings=vocab.get_vocab_size('token_target'),
                             embedding_dim=EMBEDDING_DIM)
    embedding_out = Embedding(num_embeddings=vocab.get_vocab_size('token_context'),
                              embedding_dim=EMBEDDING_DIM)

    if CUDA_DEVICE > -1:
        embedding_in = embedding_in.to(CUDA_DEVICE)
        embedding_out = embedding_out.to(CUDA_DEVICE)

    iterator = BasicIterator(batch_size=BATCH_SIZE)
    iterator.index_with(vocab)  # important, transform token to index

    model = SkipGramNegativeSamplingModel(vocab, embedding_in, embedding_out, neg_samples=10, cuda_device=CUDA_DEVICE)
    #
    # model = SkipGramModel(vocab=vocab,
    #                       embedding_in=embedding_in,
    #                       cuda_device=CUDA_DEVICE)

    optimizer = optim.Adam(model.parameters())

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=text8,
                      num_epochs=5,
                      cuda_device=CUDA_DEVICE)
    trainer.train()

    # write_embeddings(embedding_in, 'data/text8/embeddings.txt', vocab)
    print(get_synonyms('one', embedding_in, vocab))
    print(get_synonyms('december', embedding_in, vocab))
    print(get_synonyms('flower', embedding_in, vocab))
    print(get_synonyms('design', embedding_in, vocab))
    print(get_synonyms('snow', embedding_in, vocab))

    rho = evaluate_embeddings(embedding_in, vocab)
    print('simlex999 speareman correlation: {}'.format(rho))


if __name__ == '__main__':
    main()