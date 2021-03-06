import unittest
from unittest.mock import patch
import torch
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence
from pttagger.bilstm import BiLSTM
from pttagger.dataset import Dataset


class TestBiLSTM(unittest.TestCase):

    def assertTorchEqual(self, x, y):
        self.assertTrue(torch.equal(x, y))

    def setUp(self):
        self.y_to_index = {'<PAD>': 0, '<UNK>': 1, '名詞': 2, '助詞': 3,
                           '動詞': 4, '副詞': 5, '形容詞': 6}
        self.x_to_index = [{'<PAD>': 0, '<UNK>': 1, '人参': 2, 'を': 3,
                            '切る': 4, 'ざっくり': 5, '葱': 6, 'は': 7,
                            '細く': 8, '刻む': 9}]

        self.model = BiLSTM([2], [], 4, [len(self.x_to_index[0])],
                            len(self.y_to_index), batch_size=3, use_lstm=True)
        self.embedding_weight = Parameter(torch.tensor([[0, 0],  # for <PAD>
                                                        [1, 2],  # for <UNK>
                                                        [3, 4],
                                                        [5, 6],
                                                        [7, 8],
                                                        [9, 10],
                                                        [11, 12],
                                                        [13, 14],
                                                        [15, 16],
                                                        [17, 18]],
                                                       dtype=torch.float))
        self.X1 = ([2, 3, 4], [5, 4], [6, 7, 8, 9])
        self.X2 = [[6, 7, 8, 9], [2, 3, 4], [5, 4]]
        self.X3 = [[6, 7, 8, 9], [2, 3, 4, 0], [5, 4, 0, 0]]
        self.X4 = torch.tensor([[[11, 12], [13, 14], [15, 16], [17, 18]],
                                [[3, 4], [5, 6], [7, 8], [0, 0]],
                                [[9, 10], [7, 8], [0, 0], [0, 0]]],
                               dtype=torch.float)
        self.X5 = PackedSequence(torch.tensor([[11, 12],
                                               [3, 4],
                                               [9, 10],
                                               [13, 14],
                                               [5, 6],
                                               [7, 8],
                                               [15, 16],
                                               [7, 8],
                                               [17, 18]],
                                              dtype=torch.float),
                                 torch.tensor([3, 3, 2, 1]))
        self.Y = ([2, 3, 4], [5, 4], [2, 3, 6, 4])
        self.lengths = [len(x) for x in self.X2]

    def test__split(self):
        examples = [
            {'Xs': [['人参', 'を', '切る']],
             'Y': ['名詞', '助詞', '動詞']},
            {'Xs': [['ざっくり', '切る']],
             'Y': ['副詞', '動詞']},
            {'Xs': [['葱', 'は', '細く', '刻む']],
             'Y': ['名詞', '助詞', '形容詞', '動詞']}
        ]
        dataset = Dataset(examples)
        batches = self.model._split(dataset)
        self.assertEqual(batches[0], (self.X1, self.Y))

        model = BiLSTM([2], [], 4, [len(self.x_to_index[0])],
                       len(self.y_to_index), batch_size=4, use_lstm=True)
        batches = model._split(dataset)
        self.assertEqual(batches[0], (self.X1, self.Y))

    def test__sort(self):
        X2, indices = self.model._sort(self.X1)
        self.assertEqual(X2, self.X2)
        self.assertEqual(indices, [2, 0, 1])

    def test__pad(self):
        X3 = self.model._pad(self.X2, self.lengths)
        self.assertEqual(X3, self.X3)

    def test__embed(self):
        with patch.object(self.model.embeddings[0], 'weight',
                          self.embedding_weight):
            Xs = self.model._embed([torch.tensor(self.X3)])
            self.assertTorchEqual(Xs[0], self.X4)

        model = BiLSTM([2], [2], 4, [len(self.x_to_index[0]), 0],
                       len(self.y_to_index), batch_size=3, use_lstm=True)
        with patch.object(model.embeddings[0], 'weight',
                          self.embedding_weight):
            Xs = model._embed([torch.tensor(self.X3), torch.tensor(self.X3)])
            self.assertTorchEqual(Xs[0], self.X4)
            self.assertTorchEqual(Xs[1],
                                  model._unsqueeze(torch.tensor(self.X3)))

    def test__cat(self):
        X4 = torch.tensor([[[1], [2], [3], [4]], [[5], [6], [7], [8]]])
        Xs = [X4, X4]
        self.assertTrue(torch.equal(
            self.model._cat(Xs),
            torch.tensor([[[1, 1], [2, 2], [3, 3], [4, 4]],
                          [[5, 5], [6, 6], [7, 7], [8, 8]]])
        ))

    def test__pack(self):
        X5 = self.model._pack(self.X4, self.lengths)
        self.assertTorchEqual(X5.data, self.X5.data)
        self.assertTorchEqual(X5.batch_sizes, self.X5.batch_sizes)

    def test__rnn(self):
        self.model.hidden = self.model._init_hidden()
        X6, hidden = self.model._rnn(self.X5)
        # (9, 8) is length of packed sequence and dimension of hidden
        self.assertEqual(X6.data.shape, (9, 8))
        self.assertEqual(hidden[0].shape, (2, 3, 4))
        self.assertEqual(hidden[1].shape, (2, 3, 4))

    def test__unpack(self):
        self.model.hidden = self.model._init_hidden()
        X6, hidden = self.model._rnn(self.X5)
        X7 = self.model._unpack(X6)
        # batch size, sequence length, hidden size
        self.assertEqual(X7[0].data.shape, torch.Size([3, 4, 8]))
        # padded values should be zeros
        self.assertTorchEqual(X7[0].data[1][3], torch.zeros(8))
        self.assertTorchEqual(X7[0].data[2][2], torch.zeros(8))
        self.assertTorchEqual(X7[0].data[2][3], torch.zeros(8))
        # batch sizes
        self.assertTorchEqual(X7[1], torch.tensor([4, 3, 2]))

    # def test__calc_cross_entropy(self):
    #     X = torch.tensor([[[-2.02, -1.97, -1.66, -1.91, -1.51, -1.75],
    #                        [-2.06, -1.85, -1.70, -2.10, -1.47, -1.69]],
    #                       [[-2.12, -1.91, -1.65, -2.05, -1.42, -1.75],
    #                        [-2.16, -1.85, -1.66, -2.14, -1.41, -1.72]]])
    #     Y = torch.tensor([[1, 2], [1, 0]])
    #     loss = self.model._calc_cross_entropy(X, Y)
    #     self.assertTrue(torch.equal(loss, torch.tensor(1.86)))


if __name__ == '__main__':
    unittest.main()
