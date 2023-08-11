from common import *

from sklearn.metrics import f1_score
from tqdm import tqdm

import numpy as np
import pygad

# Constants
SOLUTIONS = 30
GENERATIONS = 50

# Return the predicted values
def get_predicted_values(solution, features):

    pred_y = np.clip(np.dot(features, solution), 0, 1)
    pred_y = np.where(pred_y > 0.5, 1, 0)
    return pred_y

# Define fitness function to be used by the PyGAD instance
def fitness_func(self, solution, sol_idx):

    global train_x, train_y

    pred_y = get_predicted_values(solution, train_x)
    result = f1_score(train_y, pred_y, average='binary', pos_label=1) + \
             f1_score(train_y, pred_y, average='binary', pos_label=0)

    return result

# Get the datasets to be used in the tests
train, validation, test = get_datasets()

# Get Features and Targets
train_x, train_y = get_features_targets(train)
val_x, val_y = get_features_targets(validation)
test_x, test_y = get_features_targets(test)

# Train Model
with tqdm(total=GENERATIONS) as pbar:

    # Create Genetic Algorithm
    ga_instance = pygad.GA(num_generations=GENERATIONS,
                           num_parents_mating=5,
                           fitness_func=fitness_func,
                           sol_per_pop=SOLUTIONS,
                           num_genes=BARS,
                           gene_space={'low': -1, 'high': 1},
                           random_seed=42,
                           on_generation=lambda _: pbar.update(1),
                           )

    # Run the Genetic Algorithm
    ga_instance.run()

# Set the best weights
solution, _, _ = ga_instance.best_solution()

# Predict and save train values
pred_y = get_predicted_values(solution, train_x)
save_result(train_y, pred_y, 'ga.train.csv.gz')

# Predict and save validation values
pred_y = get_predicted_values(solution, val_x)
save_result(val_y, pred_y, 'ga.val.csv.gz')

# Predict, show and save test values
pred_y = get_predicted_values(solution, test_x)
show_result(test_y, pred_y)
save_result(test_y, pred_y, 'ga.test.csv.gz')