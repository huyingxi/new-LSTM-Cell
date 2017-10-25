import torch
import torch.nn as nn
import torch.nn._functions.rnn as rnn
from torch.autograd import Variable
import argparse
from nltk import FreqDist
import sys
import string
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from gensim.models import word2vec
import os


ap = argparse.ArgumentParser()
ap.add_argument('-max_len', type=int, default=200)
ap.add_argument('-vocab_size', type=int, default=45000)
ap.add_argument('-batch_size', type=int, default=64)
ap.add_argument('-layer_num', type=int, default=1)
ap.add_argument('-hidden_dim', type=int, default=300)
ap.add_argument('-nb_epoch', type=int, default=5)
ap.add_argument('-mode', default='train')
ap.add_argument('-embed_dim', type=int, default=300)
args = vars(ap.parse_args())

MAX_LEN = args['max_len']
VOCAB_SIZE = args['vocab_size']
BATCH_SIZE = args['batch_size']
LAYER_NUM = args['layer_num']
HIDDEN_DIM = args['hidden_dim']
NB_EPOCH = args['nb_epoch']
MODE = args['mode']
EMBED_DIM = args['embed_dim']

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans


def text_to_word_sequence(text,
                          filters=' \t\n',
                          lower=False, split=" "):
    if lower:
        text = text.lower()
    text = text.translate(maketrans(filters, split * len(filters)))
    seq = text.split(split)
    return [i for i in seq if i]


def load_data(source, dist, max_len, vocab_size):
    f = open(source, 'r')
    X_data = f.read()
    f.close()
    f = open(dist, 'r')
    y_data = f.read()
    f.close()

    # Splitting raw text into array of sequences
    X = [[i for i in (x.split(' '))] for x, y in zip(X_data.split('\n'), y_data.split('\n')) if
         len(x) > 0 and len(y) > 0 and len(x.split(' ')) <= max_len and len(y.split(' ')) <= max_len]
    X_max = max(map(len,X))

    y = [[j for j in (y.split(' '))] for x, y in zip(X_data.split('\n'), y_data.split('\n')) if
         len(x) > 0 and len(y) > 0 and len(x.split(' ')) <= max_len and len(y.split(' ')) <= max_len]

    for index in range(len(X)):
        round = X_max - len(X[index])
        while(round):
            X[index].append('.')
            y[index].append('O')
            round -= 1



    model = word2vec.Word2Vec.load('/Users/test/Desktop/RE/mode.bin')

    words = list(model.wv.vocab)
    X_ix_to_word = words
    X_ix_to_word.append('UNK')
    X_word_to_ix = {word: ix for ix, word in enumerate(X_ix_to_word)}

    weight = []
    for i in range(len(X_ix_to_word)):
        if i in model.wv.vocab:
            weight.append(model[X_ix_to_word[i]])
        else:
            weight.append([np.random.randn(300, )])
    dist = FreqDist(np.hstack(y))
    y_vocab = dist.most_common(vocab_size - 1)

    count_in = 0
    count_out = 0
    for i, sentence in enumerate(X):
        for j, word in enumerate(sentence):
            if word in X_word_to_ix:
                count_in += 1
                X[i][j] = X_word_to_ix[word]
            else:
                count_out += 1
                X[i][j] = X_word_to_ix['UNK']

    y_ix_to_word = [word[0] for word in y_vocab]
    y_ix_to_word.append('UNK')
    y_word_to_ix = {word: ix for ix, word in enumerate(y_ix_to_word)}
    count_in = 0
    count_out = 0
    for i, sentence in enumerate(y):
        for j, word in enumerate(sentence):
            if word in y_word_to_ix:
                count_in += 1
                y[i][j] = y_word_to_ix[word]
            else:
                count_out += 1
                y[i][j] = y_word_to_ix['UNK']



    return (X, len(X_word_to_ix), X_word_to_ix, X_ix_to_word, y, len(y_word_to_ix), y_word_to_ix, y_ix_to_word, weight)


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, word_embed_weight):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(np.array(word_embed_weight)))

        self.lstm_addc_encoder = rnn.AutogradRNN("LSTMCell_AddC", input_size=EMBED_DIM, hidden_size=HIDDEN_DIM,
                                                 num_layers=1, batch_first=True,
                                                 dropout=0.5, train=True, bidirectional=True, batch_sizes=None,
                                                 dropout_state=None, flat_weight=None, procedure='encoder')
        self.lstm_addc_decoder = rnn.AutogradRNN("LSTMCell_AddC_decoder", input_size=HIDDEN_DIM, hidden_size=HIDDEN_DIM,
                                                 num_layers=1, batch_first=True,
                                                 dropout=0.5, train=True, bidirectional=False, batch_sizes=None,
                                                 dropout_state=None, flat_weight=None, procedure='decoder')
        self.dropout = torch.nn.Dropout(0.3)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        embeds = self.dropout(embeds)
        hidden_encoder = [
            [autograd.Variable(torch.Tensor(HIDDEN_DIM)).cuda(), autograd.Variable(torch.Tensor(HIDDEN_DIM)).cuda()],
            [autograd.Variable(torch.Tensor(HIDDEN_DIM)).cuda(), autograd.Variable(torch.Tensor(HIDDEN_DIM)).cuda()],
        ]

        for i in range(2):
            for j in range(2):
                hidden_encoder[i][j] = hidden_encoder[i][j]
                hidden_encoder[i][j].data.uniform_()
        weight_encoder = [
            [torch.Tensor(EMBED_DIM, HIDDEN_DIM * 4), torch.Tensor(HIDDEN_DIM, HIDDEN_DIM * 4),
             torch.Tensor(HIDDEN_DIM, HIDDEN_DIM * 4)],
            [torch.Tensor(EMBED_DIM, HIDDEN_DIM * 4), torch.Tensor(HIDDEN_DIM, HIDDEN_DIM * 4),
             torch.Tensor(HIDDEN_DIM, HIDDEN_DIM * 4)]
        ]
        for i in range(2):
            for j in range(3):
                weight_encoder[i][j] = Variable(weight_encoder[i][j], requires_grad=True)
                weight_encoder[i][j].data.uniform_()
        hidden_decoder = [
            [autograd.Variable(torch.Tensor(HIDDEN_DIM)).cuda()],  # hidden_h
            [autograd.Variable(torch.Tensor(HIDDEN_DIM)).cuda()],  # hidden_c
            [autograd.Variable(torch.Tensor(HIDDEN_DIM)).cuda()],  # hidden_o
        ]

        for i in range(3):
            hidden_decoder[i][0] = hidden_decoder[i][0]
            hidden_decoder[i][0].data.uniform_()
        weight_decoder = [
            [
                torch.Tensor(EMBED_DIM, HIDDEN_DIM * 4),
                torch.Tensor(HIDDEN_DIM, HIDDEN_DIM * 4),
                torch.Tensor(HIDDEN_DIM, HIDDEN_DIM * 4),
                torch.Tensor(HIDDEN_DIM, HIDDEN_DIM * 4),
                torch.Tensor(HIDDEN_DIM, HIDDEN_DIM),
            ]
        ]
        for i in range(1):
            for j in range(5):
                weight_decoder[i][j] = Variable(weight_decoder[i][j].cuda())
                weight_decoder[i][j].data.uniform_()
        lstm_out, hidden_encoder = self.lstm_addc_encoder(
            input=embeds.view(BATCH_SIZE, len(sentence[0]), -1), weight=weight_encoder, hidden=hidden_encoder)

        hidden_decoder_input = lstm_out  # [ (1, 15, 300)]

        lstm_out_decoder, hidden_decoder = self.lstm_addc_decoder(
            input=hidden_decoder_input, weight=weight_decoder, hidden=hidden_decoder)
        tag_space = self.hidden2tag(lstm_out_decoder)
        tag_scores = F.log_softmax(tag_space)

        return tag_scores


def predict(X, y, model, loss_function):
    model.zero_grad()
    sentence = np.asarray(X)
    tensor = torch.from_numpy(sentence)
    sentence_in = autograd.Variable(tensor).cuda()
    tags = np.asarray(y)
    targets = torch.from_numpy(tags)
    targets_in = autograd.Variable(targets).cuda()
    tag_scores = model(sentence_in)
    loss = 0
    for i in range(len(sentence)):
        loss += loss_function(tag_scores[i], targets_in[i])
    return loss/len(sentence)


def load_test(file1,file2,X_word_to_ix,y_word_to_ix):
    f = open(file1, 'r')
    f1 = open(file2, 'r')
    X_test_data = f.read()
    Y_test_data = f1.read()
    f.close()
    f1.close()
    test_x = [text_to_word_sequence(x_)[::-1] for x_ in X_test_data.split('\n') if
              len(x_.split(' ')) > 0 and len(x_.split(' ')) <= MAX_LEN]
    test_y = [text_to_word_sequence(y_)[::-1] for y_ in Y_test_data.split('\n') if
              len(y_.split(' ')) > 0 and len(y_.split(' ')) <= MAX_LEN]

    X_max_test = max(map(len, test_x))
    for index in range(len(test_x)):
        round = X_max_test - len(test_x[index])
        while (round):
            test_x[index].append('.')
            test_y[index].append('O')
            round -= 1

    for i, sentence in enumerate(test_x):
        for j, word in enumerate(sentence):
            if word in X_word_to_ix:
                test_x[i][j] = X_word_to_ix[word]
            else:
                test_x[i][j] = X_word_to_ix['UNK']

    for i, sentence in enumerate(test_y):
        for j, word in enumerate(sentence):
            if word in y_word_to_ix:
                test_y[i][j] = y_word_to_ix[word]
            else:
                test_y[i][j] = y_word_to_ix['UNK']

    return test_x, test_y

def run():
    X, X_vocab_len, X_word_to_ix, X_ix_to_word, y, y_vocab_len, y_word_to_ix, y_ix_to_word, word_embed_weight = load_data(
        '/Users/test/Desktop/RE/data/train_test/train_x_real.txt', '/Users/test/Desktop/RE/data/train_test/train_y_real.txt', MAX_LEN, VOCAB_SIZE)
    model = LSTMTagger(EMBED_DIM, HIDDEN_DIM, len(X_word_to_ix), len(y_word_to_ix), word_embed_weight)
    model.cuda()
    loss_function = nn.NLLLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0,
                              centered=False)

    test_x, test_y = load_test('/Users/test/Desktop/RE/data/train_test/test_x_real.txt','/Users/test/Desktop/RE/data/train_test/test_y_real.txt',
                               X_word_to_ix,y_word_to_ix)

    count = 0
    for epoch in xrange(NB_EPOCH):  # again, normally you would NOT do 300 epochs, it is toy data
        for i in range(len(X) - BATCH_SIZE):
            loss = predict(X[i:i+BATCH_SIZE], y[i:i+BATCH_SIZE], model, loss_function)
            loss.backward()
            optimizer.step()
            if count % 1000 == 0:
                # torch.save(model, '/Users/test/Desktop/RE/model')
                print("{0} epoch , training loss {1} : ".format(epoch,loss))
                loss_test = 0
                loss_test += predict(test_x[0:BATCH_SIZE], test_y[0:BATCH_SIZE], model, loss_function)
                print("test loss {} : ".format(loss_test/BATCH_SIZE))
            count += 1

run()


