import tensorflow as tf
import pandas as pd
# import numpy as np

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dftest = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

ytrain = dftrain.pop('survived')
print(dftrain.describe())
yeval = dftest.pop('survived')

cat_columns = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']

num_columns = ['age', 'fare']

feature_cols = []
for feature_name in cat_columns:
    vocab = dftrain[feature_name].unique()
    feature_cols.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab))

for feature_name in num_columns:
    feature_cols.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_cols)

