========
pttagger
========

.. image:: https://img.shields.io/pypi/v/pttagger.svg
        :target: https://pypi.python.org/pypi/pttagger

.. image:: https://img.shields.io/travis/jun-harashima/pttagger.svg
        :target: https://travis-ci.org/jun-harashima/pttagger

pttagger is a simple PyTorch-based tagger which has the following features:

- bi-directional LSTM
- variable-sized mini-batches
- multiple inputs

Quick Start
===========

Installation
------------

Run this command in your terminal:

.. code-block:: bash

   $ pip install pttagger

Pre-processing
--------------

Suppose that you have the following examples of named entity recognition:

   Joe/B-PER Smith/I-PER goes/O to/O Japan/B-LOC ./O
   Jane/B-PER Smith/I-PER belongs/O to/O Kyoto/B-ORG University/I-ORG ./O
   ...

First, give the examples to construct a ``Dataset`` object like this:

.. code-block:: python

   examples = [
       {'Xs': [['Joe', 'Doe', 'goes', 'to', 'Japan', '.']],
        'Y': ['B-PER', 'I-PER', 'O', 'O', 'B-LOC', 'O']},
       {'Xs': [['Jane', 'Doe', 'belongs', 'to', 'Kyoto', 'University', '.']],
        'Y': ['B-PER', 'I-PER', 'O', 'O', 'B-ORG', 'I-ORG', 'O']},
       ...
   ]
   dataset = Dataset(examples)

You can also use multiple inputs as a value of ``Xs``. In the following case, ``Xs`` has not only word information but also POS information:

   examples = [
       {'Xs': [['Joe', 'Doe', 'goes', 'to', 'Japan', '.'], ['NNP', 'NNP', 'VBZ', 'TO', 'NNP', '.']],
        'Y': ['B-PER', 'I-PER', 'O', 'O', 'B-LOC', 'O']},
       {'Xs': [['Jane', 'Doe', 'belongs', 'to', 'Kyoto', 'University', '.'], ['NNP', 'NNP', 'VBZ', 'TO', 'NNP', 'NNP', '.']],
        'Y': ['B-PER', 'I-PER', 'O', 'O', 'B-ORG', 'I-ORG', 'O']},
       ...
   ]
   dataset = Dataset(examples)

Now, ``dataset`` has the following two indices:

- ``x_to_index``: e.g., [{'<PAD>': 0, '<UNK>': 1, 'Joe': 2, 'Doe': 3, ...}]
- ``y_to_index``: e.g., {'<PAD>': 0, '<UNK>': 1, 'B-PER': 2, 'I-PER': 3, ...}

If you use multiple inputs, ``x_to_index`` has indices for each input.

Training
--------

Construct a ``Model`` object and train it as follows:

.. code-block:: python

   EMBEDDING_DIMS = [100]  # if you use multiple inputs, set a dimension for each input
   HIDDEN_DIMS = [10]  # the same as above
   x_set_sizes = [len(x_to_index) for x_to_index in dataset.x_to_index]
   y_set_size = len(dataset.y_to_index)
   model = Model(EMBEDDING_DIMS, HIDDEN_DIMS, x_set_sizes, y_set_size)
   model.train(dataset)

Test
----

Predict tags for test examples like this:

.. code-block:: python

   test_examples = [
       {'Xs': [['Richard', 'Roe', 'comes', 'to', 'America', '.']],
        'Y': ['B-PERSON', 'I-PERSON', 'O', 'O', 'B-LOCATION', 'O']}
   ]
   test_dataset = Dataset(test_examples)
   results = model.test(dataset)
