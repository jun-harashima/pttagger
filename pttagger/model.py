import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
import torch.optim as optim
from pttagger.base import Base


torch.manual_seed(1)


class Model(Base):

    # For simplicity, use the same pad_index for Xs[0], Xs[1], ..., and Y
    def __init__(self, embedding_dims, nonembedding_dims, hidden_dims,
                 x_set_sizes, y_set_size, pad_index=0, batch_size=16,
                 use_lstm=False, num_layers=1):
        super(Model, self).__init__()
        self.embedding_dims = embedding_dims
        self.nonembedding_dims = nonembedding_dims
        self.hidden_dims = hidden_dims
        self.x_set_sizes = x_set_sizes
        self.y_set_size = y_set_size
        self.pad_index = pad_index
        self.batch_size = batch_size
        self.use_lstm = use_lstm
        self.num_layers = num_layers
        self.use_cuda = self._init_use_cuda()
        self.device = self._init_device()
        self.embeddings = self._init_embeddings()
        self.rnn = self._init_rnn()
        self.final = self._init_final(sum(self.hidden_dims) * 2)

    def _init_rnn(self):
        if self.use_lstm:
            return self._init_lstm()
        return self._init_gru()

    def _init_lstm(self):
        lstm = nn.LSTM(sum(self.embedding_dims) + sum(self.nonembedding_dims),
                       sum(self.hidden_dims), num_layers=self.num_layers,
                       bidirectional=True)
        return lstm.cuda() if self.use_cuda else lstm

    def _init_gru(self):
        gru = nn.GRU(sum(self.embedding_dims) + sum(self.nonembedding_dims),
                     sum(self.hidden_dims), num_layers=self.num_layers,
                     bidirectional=True)
        return gru.cuda() if self.use_cuda else gru

    def _init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        zeros = torch.zeros(self.num_layers * 2, self.batch_size,
                            sum(self.hidden_dims), device=self.device)
        if self.use_lstm:
            return (zeros, zeros)
        return zeros

    def forward(self, Xs, lengths):
        self.hidden = self._init_hidden()
        Xs = self._embed(Xs)
        X = self._cat(Xs)
        X = self._pack(X, lengths)
        X, self.hidden = self._rnn(X)
        X, _ = self._unpack(X)
        X = X.contiguous().view(-1, X.shape[2])
        # Note that the final layer returns values also for padded elements. We
        # ignore them when computing our loss.
        X = self.final(X)
        X = F.log_softmax(X, dim=1)
        X = X.view(self.batch_size, lengths[0], self.y_set_size)
        return X

    def _embed(self, Xs):
        length = len(self.embeddings)
        return [self.embeddings[i](X) if i < length else self._unsqueeze(X)
                for i, X in enumerate(Xs)]

    def _unsqueeze(self, X):
        return torch.unsqueeze(X.to(torch.float), 2)

    def _pack(self, X, lengths):
        return U.rnn.pack_padded_sequence(X, lengths, batch_first=True)

    def _rnn(self, X):
        return self.rnn(X, self.hidden)

    def _unpack(self, X):
        return U.rnn.pad_packed_sequence(X, batch_first=True)
