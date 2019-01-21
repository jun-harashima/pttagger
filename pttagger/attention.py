import torch
import torch.nn as nn
import torch.nn.functional as F
from pttagger.base import Base


torch.manual_seed(1)


class SelfAttention(Base):

    def __init__(self, embedding_dims, nonembedding_dims, hidden_dim,
                 x_set_sizes, y_set_size, pad_index=0, batch_size=16):
        super(SelfAttention, self).__init__(embedding_dims, nonembedding_dims,
                                            hidden_dim, x_set_sizes,
                                            y_set_size, pad_index=pad_index,
                                            batch_size=batch_size)
        self.query = self._init_query()
        self.key = self._init_key()
        self.value = self._init_value()
        self.attention_output = self._init_attention_output()
        self.final = self._init_final(self.hidden_dim)

    def _init_query(self):
        query = nn.Linear(sum(self.embedding_dims), self.hidden_dim)
        return query.cuda() if self.use_cuda else query

    def _init_key(self):
        return self._init_query()

    def _init_value(self):
        return self._init_query()

    def _init_attention_output(self):
        attention_output = nn.Linear(self.hidden_dim, self.hidden_dim)
        return attention_output.cuda() if self.use_cuda else attention_output

    def forward(self, Xs, lengths):
        Xs = self._embed(Xs)
        X = self._cat(Xs)
        Q = self.query(X)
        K = self.key(X)
        V = self.value(X)
        K = torch.transpose(K, 1, 2)
        logit = torch.bmm(Q, K)
        attention_weight = F.softmax(logit, dim=2)
        X = torch.bmm(attention_weight, V)
        # Note that the final layer returns values also for padded elements. We
        # ignore them when computing our loss.
        X = X.view(-1, X.shape[2])
        X = self.attention_output(X)
        X = self.final(X)
        X = F.log_softmax(X, dim=1)
        X = X.view(self.batch_size, lengths[0], self.y_set_size)
        return X
