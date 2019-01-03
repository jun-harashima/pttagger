import torch
from pttagger.model import Model
from pttagger.dataset import Dataset


EMBEDDING_DIMS = [100]
HIDDEN_DIMS = [10]

examples = [
    {'Xs': [['Joe', 'Doe', 'goes', 'to', 'Japan', '.']],
     'Y': ['B-PER', 'I-PER', 'O', 'O', 'B-LOC', 'O']},
    {'Xs': [['Jane', 'Doe', 'belongs', 'to', 'Kyoto', 'University', '.']],
     'Y': ['B-PER', 'I-PER', 'O', 'O', 'B-ORG', 'I-ORG', 'O']}
]
dataset = Dataset(examples)

x_set_sizes = [len(x_to_index) for x_to_index in dataset.x_to_index]
y_set_size = len(dataset.y_to_index)
model = Model(EMBEDDING_DIMS, HIDDEN_DIMS, x_set_sizes, y_set_size,
              batch_size=3)
model.train(dataset)
torch.save(model.state_dict(), 'ner.model')

model.load_state_dict(torch.load('ner.model'))
test_examples = [
    {'Xs': [['Richard', 'Roe', 'comes', 'to', 'America', '.']],
     'Y': ['B-PER', 'I-PER', 'O', 'O', 'B-LOC', 'O']}
]
test_dataset = Dataset(test_examples)
results = model.test(test_dataset)
print(results)
