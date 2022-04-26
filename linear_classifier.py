import tensorflow as tf
import pandas as pd
# import numpy as np

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

ytrain = dftrain.pop('survived')
print(dftrain.describe())
yeval = dfeval.pop('survived')

cat_columns = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']

num_columns = ['age', 'fare']

feature_cols = []
#Looping through the categorical features and creates a vocabulary (values that the features can take, this is done so
# that the model recognizes the possible outcomes for that feature

for feature_name in cat_columns:
    vocab = dftrain[feature_name].unique()
    feature_cols.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab))

for feature_name in num_columns:
    feature_cols.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# wrapper function for returning the input function, which is passed to the linear classified model
def make_input_fn(data_df, label_df, epochs=10, shuffle=True, batch_size=32):
    def input_fn():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))

        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(epochs)
        return ds
    return input_fn()

train_input_fn = make_input_fn(dftrain, ytrain)  # makes the input function for the test set
eval_input_fn = make_input_fn(dfeval, yeval, epochs=1, shuffle=False)  # Makes the input funtion for the test set, which
# should be run just once (epoch=1) and no shuffle is necessary.

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_cols)  # Creates a linear classified object with
# the features we have

linear_est.train(train_input_fn)  #training the model, which receives the train input funtion as argument
result = linear_est.evaluate(eval_input_fn)  # dict containing the results of the model (accuracy, etc)

# clear_output()  # clears out the console

print(result['accuracy'])


print(feature_cols)

