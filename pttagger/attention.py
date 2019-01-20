import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
import torch.optim as optim
from pttagger.base import Base


torch.manual_seed(1)


class SelfAttention(Base):

    EPOCH_NUM = 100

    def __init__(self, embedding_dims, hidden_dim, x_set_sizes, y_set_size, pad_index=0, batch_size=16):
        super(SelfAttention, self).__init__()
        self.embedding_dims = embedding_dims
        self.hidden_dim = hidden_dim
        self.x_set_sizes = x_set_sizes
        self.y_set_size = y_set_size
        self.pad_index = pad_index
        self.batch_size = batch_size
        self.use_cuda = self._init_use_cuda()
        self.device = self._init_device()
        self.embeddings = self._init_embeddings()
        self.final = self._init_final(sum(self.embedding_dims))
        self.q = self._init_q()
        self.k = self._init_q()
        self.v = self._init_q()
        self.output = self._init_output()

    def _init_q(self):
        q = nn.Linear(sum(self.embedding_dims), self.hidden_dim)
        return q.cuda() if self.use_cuda else q

    def _init_output(self):
        output = nn.Linear(self.hidden_dim, self.hidden_dim)
        return output.cuda() if self.use_cuda else output

    def forward(self, Xs, lengths):
        Xs = self._embed(Xs)
        X = self._cat(Xs)
        Q = self.q(X)
        K = self.k(X)
        V = self.v(X)
        K = torch.transpose(K, 1, 2)
        logit = torch.bmm(Q, K)
        attention_weight = F.softmax(logit, dim=2)
        X = torch.bmm(attention_weight, V)
        # Note that the final layer returns values also for padded elements. We
        # ignore them when computing our loss.
        X = X.view(-1, X.shape[2])
        X = self.output(X)
        X = self.final(X)
        X = F.log_softmax(X, dim=1)
        X = X.view(self.batch_size, lengths[0], self.y_set_size)
        return X

    def _embed(self, Xs):
        return [self.embeddings[i](X) for i, X in enumerate(Xs)]
