import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('BootcampSession1/Model/prepared_data.csv')
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.9513211014116288
exported_pipeline = KNeighborsClassifier(n_neighbors=32, p=2, weights="distance")

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
