from common import *

from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

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
model = xgb.XGBClassifier(
    n_estimators=10000,
    seed=42,
    early_stopping_rounds=10)

params = {
        'min_child_weight': [50, 100, 300, 500],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [4, 5, 6, 7, 8],
        'eta': [0.01, 0.05, 0.1]}

grid = RandomizedSearchCV(
    model,
    param_distributions=params,
    n_iter=30,
    scoring='roc_auc',
    n_jobs=4,
    random_state=42)

grid.fit(
    train_x,
    train_y,
    eval_set=[(train_x, train_y), (val_x, val_y)],
    verbose=False)

print(grid.best_params_)

# Train Model
model = xgb.XGBClassifier(
    max_depth=grid.best_params_['max_depth'],
    n_estimators=10000,
    min_child_weight=grid.best_params_['min_child_weight'],
    colsample_bytree=grid.best_params_['colsample_bytree'],
    subsample=grid.best_params_['subsample'],
    eta=grid.best_params_['eta'],
    seed=42,
    early_stopping_rounds=10)

model.fit(
    train_x,
    train_y,
    eval_set=[(train_x, train_y), (val_x, val_y)],
    verbose=True)

# Predict and save train values
pred_y = get_predicted_values(model, train_x)
save_result(train_y, pred_y, 'xgb.train.csv.gz')

# Predict and save validation values
pred_y = get_predicted_values(model, val_x)
save_result(val_y, pred_y, 'xgb.val.csv.gz')

# Predict, show and save test values
pred_y = get_predicted_values(model, test_x)
show_result(test_y, pred_y)
save_result(test_y, pred_y, 'xgb.test.csv.gz')
