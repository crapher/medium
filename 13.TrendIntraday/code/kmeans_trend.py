from common import *
from sklearn.cluster import KMeans

# Return the predicted values
def get_predicted_values(model, features):

    pred_y = model.predict(features)
    
    # Transform 0s to 1s and 1s to 0s (In this example we have the values in the opposite site)
    pred_y = ~pred_y + 2

    return pred_y

# Get the datasets to be used in the tests
train, validation, test = get_datasets()

# Get Features and Targets
train_x, train_y = get_features_targets(train)
val_x, val_y = get_features_targets(validation)
test_x, test_y = get_features_targets(test)

# Train Model
model = KMeans(n_clusters=2, random_state=42, n_init="auto")
model.fit(train_x)

# Predict and save train values
pred_y = get_predicted_values(model, train_x)
save_result(train_y, pred_y, 'kmeans.train.csv.gz')

# Predict and save validation values
pred_y = get_predicted_values(model, val_x)
save_result(val_y, pred_y, 'kmeans.val.csv.gz')

# Predict, show and save test values
pred_y = get_predicted_values(model, test_x)
show_result(test_y, pred_y)
save_result(test_y, pred_y, 'kmeans.test.csv.gz')
