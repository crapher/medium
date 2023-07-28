from common import *

# Return the predicted values
def get_predicted_values(features):

    pred_y = []

    for feature in features:
        trend = np.where(feature[0] < feature[-1], 1, 0)
        pred_y.append(trend)

    return pred_y

# Get the datasets to be used in the tests
train, validation, test = get_datasets()

# Get Features and Targets
train_x, train_y = get_features_targets(train, False)
val_x, val_y = get_features_targets(validation, False)
test_x, test_y = get_features_targets(test, False)

# Predict and save train values
pred_y = get_predicted_values(train_x)
save_result(train_y, pred_y, 'stock.train.csv.gz')

# Predict and save validation values
pred_y = get_predicted_values(val_x)
save_result(val_y, pred_y, 'stock.val.csv.gz')

# Predict, show, and save test values
pred_y = get_predicted_values(test_x)
show_result(test_y, pred_y)
save_result(test_y, pred_y, 'stock.test.csv.gz')
