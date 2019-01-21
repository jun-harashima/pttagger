import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
from pttagger.base import Base


torch.manual_seed(1)


class Model(Base):

    # For simplicity, use the same pad_index for Xs[0], Xs[1], ..., and Y
    def __init__(self, embedding_dims, nonembedding_dims, hidden_dim,
                 x_set_sizes, y_set_size, pad_index=0, batch_size=16,
                 use_lstm=False, num_layers=1):
        super(Model, self).__init__(embedding_dims, nonembedding_dims,
                                    hidden_dim, x_set_sizes, y_set_size,
                                    pad_index=pad_index, batch_size=batch_size)
        self.use_lstm = use_lstm
        self.num_layers = num_layers
        self.rnn = self._init_rnn()
        self.final = self._init_final(self.hidden_dim * 2)

    def _init_rnn(self):
        if self.use_lstm:
            return self._init_lstm()
        return self._init_gru()

    def _init_lstm(self):
        lstm = nn.LSTM(sum(self.embedding_dims) + sum(self.nonembedding_dims),
                       self.hidden_dim, num_layers=self.num_layers,
                       bidirectional=True)
        return lstm.cuda() if self.use_cuda else lstm

    def _init_gru(self):
        gru = nn.GRU(sum(self.embedding_dims) + sum(self.nonembedding_dims),
                     self.hidden_dim, num_layers=self.num_layers,
                     bidirectional=True)
        return gru.cuda() if self.use_cuda else gru

    def _init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        zeros = torch.zeros(self.num_layers * 2, self.batch_size,
                            self.hidden_dim, device=self.device)
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

    def _pack(self, X, lengths):
        return U.rnn.pack_padded_sequence(X, lengths, batch_first=True)

    def _rnn(self, X):
        return self.rnn(X, self.hidden)

    def _unpack(self, X):
        return U.rnn.pad_packed_sequence(X, batch_first=True)
