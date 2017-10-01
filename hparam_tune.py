# Enable future and backwards compatability:
from __future__ import absolute_import, division, generators, unicode_literals, print_function, nested_scopes, with_statement
import word2vec


### Run the word to vec script for hyperparameter tuning of the word2vec model ###

##
## @brief      Makes a hparam string for labeling training runs.
##
## @param      learning_rate  The learning rate
## @param      skip_window    The skip window
## @param      batch_size     The batch size
##
## @return     { A string that will be used to name the training run }
##
def make_hparam_string(learning_rate, skip_window, batch_size):
  string = 'lr=' + str(learning_rate) + "_" + 'skipW=' + str(skip_window) + '_' + 'batch=' + str(batch_size)
  return string

def init():
  # Try some learning rates
  for learning_rate in [0.01, 0.03, 0.09, 0.1, 0.3, 0.9]:
    # Try some different size skip windows
    for skip_window in [2, 3, 4]:
      # Try some different input reuse values
      for batch_size in [128, 192, 256]:
      #   # Construct a hyperparameter string for each one e.g. lr=0.03, skip_window=1,
      #   #  batch_size = 128
        hparam_str = make_hparam_string(learning_rate, skip_window, batch_size)
        # Run the model with various hyperparameters
        word2vec.init(learning_rate, skip_window, batch_size, hparam_str)

if __name__ == '__main__':
  init()
