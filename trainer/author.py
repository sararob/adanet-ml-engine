# Copyright 2019 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import urllib

import adanet
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow_hub as hub

# Define some globals
model_dir='gs://path/to/your/gcs/folder'
batch_size=64
total_steps=40000

# Get the data
urllib.request.urlretrieve('https://storage.googleapis.com/authors-training-data/data.csv', 'data.csv')

data = pd.read_csv('data.csv')
data = data.sample(frac=1)
data.head()


# Split into train and test sets
train_size = int(len(data) * .8)

train_text = data['text'][:train_size]
train_authors = data['author'][:train_size]

test_text = data['text'][train_size:]
test_authors = data['author'][train_size:]


# Turn the labels into a one-hot encoding
encoder = LabelEncoder()
encoder.fit_transform(np.array(train_authors))
train_encoded = encoder.transform(train_authors)
test_encoded = encoder.transform(test_authors)
print(encoder.classes_)
num_classes = len(encoder.classes_)


# Create TF Hub embedding columns using 2 different modules
ndim_embeddings = hub.text_embedding_column(
  "ndim",
  module_spec="https://tfhub.dev/google/nnlm-en-dim128/1", trainable=False 
)
encoder_embeddings = hub.text_embedding_column(
  "encoder", 
  module_spec="https://tfhub.dev/google/universal-sentence-encoder/2", trainable=False)


# Create a head and features dict for training
multi_class_head = tf.contrib.estimator.multi_class_head(
  len(encoder.classes_),
  loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE
)

train_features = {
  "ndim": train_text,
  "encoder": train_text
}

train_labels = np.array(train_encoded).astype(np.int32)

# Train input function
def input_fn_train():
  dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
  dataset = dataset.repeat().shuffle(100).batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  data, labels = iterator.get_next()
  return data, labels

# Define the Estimators we'll be feeding into our AdaNet model
estimator_ndim = tf.contrib.estimator.DNNEstimator(
  head=multi_class_head,
  hidden_units=[64,10],
  feature_columns=[ndim_embeddings]
)

estimator_encoder = tf.contrib.estimator.DNNEstimator(
  head=multi_class_head,
  hidden_units=[64,10],
  feature_columns=[encoder_embeddings]
)


# Create our AutoEnsembleEstimator from the 2 estimators above
estimator = adanet.AutoEnsembleEstimator(
    head=multi_class_head,
    candidate_pool=[
        estimator_encoder,
        estimator_ndim
    ],
    config=tf.estimator.RunConfig(
      save_summary_steps=1000,
      save_checkpoints_steps=1000,
      model_dir=model_dir
    ),
    max_iteration_steps=5000)


# Set up features dict and input function for eval
eval_features = {
  "ndim": test_text,
  "encoder": test_text
}

eval_labels = np.array(test_encoded).astype(np.int32)

def input_fn_eval():
  dataset = tf.data.Dataset.from_tensor_slices((eval_features, eval_labels))
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  data, labels = iterator.get_next()
  return data, labels


# Configurations for running train_and_evaluate
train_spec = tf.estimator.TrainSpec(
  input_fn=input_fn_train,
  max_steps=total_steps
)

eval_spec=tf.estimator.EvalSpec(
  input_fn=input_fn_eval,
  steps=None,
  start_delay_secs=10,
  throttle_secs=10
)


tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


# Create serving input function to be able to serve predictions later using provided inputs
def serving_input_fn():
    feature_placeholders = {
      'encoder' : tf.placeholder(tf.string, [None]),
      'ndim' : tf.placeholder(tf.string, [None])
    }

    return tf.estimator.export.ServingInputReceiver(feature_placeholders, feature_placeholders)


latest_ckpt = tf.train.latest_checkpoint(model_dir)
last_eval = estimator.evaluate(
  input_fn_eval,
  checkpoint_path=latest_ckpt
)

# Export the model to GCS for serving
exporter = tf.estimator.LatestExporter('exporter', serving_input_fn, exports_to_keep=None)
exporter.export(estimator, model_dir, latest_ckpt, last_eval, is_the_final_export=True)