# word2vec_example
An example of a tensorflow word2vec model with a script for hyperparameter tuning and configuration for tensorboard

Requires that you create your own data preprocessing script that outputs a string list and dictionary with the index of the words as keys and the actual words as values

Run the model with word2vec.py
The model name will be saved as '1' in the saved_models directory

Run the model with hparam_tune.py to see the effect of various hyperparameter settings on the model
Various model names will be saved using identifyers for their parameter values in the saved_models directory

To run tensorboard for one model:
tensorboard --logdir ./tmp/<model-name>

To run tensorboard to compare models
tensorboard --logdir ./tmp/

Python packages used:
- os
- numpy
- collections
- random
- math
- tensorflow
