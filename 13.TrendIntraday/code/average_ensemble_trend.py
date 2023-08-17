from common import *

import os
import glob
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import confusion_matrix, classification_report

# Return the predicted values
def get_predicted_values(features):
    
    pred_y = np.where(features.sum(axis=1) > features.shape[1] / 2, 1, 0)
    return pred_y
    
# Get return a dataset with all the level 1 results
def generate_features_targets(files_pattern):

    files = glob.glob(files_pattern)

    x = None
    y = None
    for file in files:
        col_name = os.path.basename(file).split('.')[0]
        tmp_df = pd.read_csv(file)
        tmp_df.columns = [col_name, 'target']
        
        if y is None:
            y = tmp_df['target']
            x = tmp_df[[col_name]]
        else:
            x[col_name] = tmp_df[col_name]

    return x.values, y.values

# Get the datasets to be used in the tests
train_x, train_y = generate_features_targets('../result/*train.csv.gz')
val_x, val_y = generate_features_targets('../result/*val.csv.gz')
test_x, test_y = generate_features_targets('../result/*test.csv.gz')

# Predict and show train values
pred_y = get_predicted_values(train_x)
show_result(train_y, pred_y, 'TRAIN')

# Predict and show validation values
pred_y = get_predicted_values(val_x)
show_result(val_y, pred_y, 'VALIDATION')

# Predict and show test values
pred_y = get_predicted_values(test_x)
show_result(test_y, pred_y)
