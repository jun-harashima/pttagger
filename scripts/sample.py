import torch
from pttagger.model import Model
from pttagger.dataset import Dataset


EMBEDDING_DIMS = [2, 2]
HIDDEN_DIMS = [4, 4]

examples = [
    {'Xs': [['人参', 'を', '切る'], ['名詞', '助詞', '動詞']],
     'Y': ['B-Food', 'O', 'B-Action']},
    {'Xs': [['ざっくり', '切る'], ['副詞', '動詞']],
     'Y': ['O', 'B-Action']},
    {'Xs': [['九条', '葱', 'は', '刻む'], ['名詞', '名詞', '助詞', '動詞']],
     'Y': ['B-Food', 'I-Food', 'O', 'B-Action']}
]
dataset = Dataset(examples)

x_set_sizes = [len(x_to_index) for x_to_index in dataset.x_to_index]
y_set_size = len(dataset.y_to_index)
model = Model(EMBEDDING_DIMS, HIDDEN_DIMS, x_set_sizes, y_set_size,
              batch_size=3)
model.train(dataset)
torch.save(model.state_dict(), 'ner.model')

model.load_state_dict(torch.load('ner.model'))
examples = [
    {'Xs': [['葱', 'を', '切る'], ['名詞', '助詞', '動詞']],
     'Y': ['B-Food', 'O', 'B-Action']},
    {'Xs': [['細く', '切る'], ['副詞', '動詞']],
     'Y': ['O', 'B-Action']},
    {'Xs': [['三浦', '大根', 'は', '刻む'], ['名詞', '名詞', '助詞', '動詞']],
     'Y': ['B-Food', 'I-Food', 'O', 'B-Action']}
]
dataset = Dataset(examples)
results = model.test(dataset)
print(results)
