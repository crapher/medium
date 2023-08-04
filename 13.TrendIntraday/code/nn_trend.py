from common import *

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.utils import set_random_seed

# Return the predicted values
def get_predicted_values(model, features):

    pred_y = model.predict(features)
    pred_y = pred_y.flatten()
    pred_y = np.where(pred_y > 0.5, 1, 0)
    return pred_y

# Get the datasets to be used in the tests
train, validation, test = get_datasets()

# Get Features and Targets
train_x, train_y = get_features_targets(train)
val_x, val_y = get_features_targets(validation)
test_x, test_y = get_features_targets(test)

# Train Model
set_random_seed(42)                         # Allow reproducibility
early_stopping = EarlyStopping(patience=2)  # Stop if the model does not improve

model = Sequential()
model.add(Dense(16, input_dim=15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=100, batch_size=32, validation_data=(val_x, val_y), callbacks=[early_stopping])

# Predict and save train values
pred_y = get_predicted_values(model, train_x)
save_result(train_y, pred_y, 'nn.train.csv.gz')

# Predict and save validation values
pred_y = get_predicted_values(model, val_x)
save_result(val_y, pred_y, 'nn.val.csv.gz')

# Predict, show and save test values
pred_y = get_predicted_values(model, test_x)
show_result(test_y, pred_y)
save_result(test_y, pred_y, 'nn.test.csv.gz')