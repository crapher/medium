from common import *

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

import numpy as np

# Return the predicted values
def get_predicted_values(model, features):

    pred_y = model.predict(features)
    return pred_y

# Get the datasets to be used in the tests
train, validation, test = get_datasets()

# Get Features and Targets
train_x, train_y = get_features_targets(train)
val_x, val_y = get_features_targets(validation)
test_x, test_y = get_features_targets(test)

# Find the best parameters
model = KNeighborsClassifier()

params = {
        'n_neighbors': list(range(3,300)),
        'weights': ['uniform', 'distance']
        }

grid = GridSearchCV(
    model,
    param_grid=params,
    scoring='roc_auc',
    n_jobs=4)

grid.fit(
    np.concatenate((train_x, val_x)),
    np.concatenate((train_y, val_y)))

print(grid.best_params_)

# Train Model
model = KNeighborsClassifier(
    n_neighbors=grid.best_params_['n_neighbors'],
    weights=grid.best_params_['weights'])

model.fit(
    np.concatenate((train_x, val_x)),
    np.concatenate((train_y, val_y)))

# Predict and save train values
pred_y = get_predicted_values(model, train_x)
save_result(train_y, pred_y, 'knn.train.csv.gz')

# Predict and save validation values
pred_y = get_predicted_values(model, val_x)
save_result(val_y, pred_y, 'knn.val.csv.gz')

# Predict, show and save test values
pred_y = get_predicted_values(model, test_x)
show_result(test_y, pred_y)
save_result(test_y, pred_y, 'knn.test.csv.gz')
