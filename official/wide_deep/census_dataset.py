# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using tf.estimator API."""
# refereces : https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/tutorials/wide_and_deep/
#             http://bcho.tistory.com/tag/wide%20and%20deep%20model
"""Download and clean the Census Income Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# pylint: disable=wrong-import-order
from absl import app as absl_app
from absl import flags
from six.moves import urllib
import tensorflow as tf
# pylint: enable=wrong-import-order

from official.utils.flags import core as flags_core


DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult'
TRAINING_FILE = 'adult.data'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
EVAL_FILE = 'adult.test'
EVAL_URL = '%s/%s' % (DATA_URL, EVAL_FILE)


# tf.enable_eager_execution()
from tensorflow.python import debug as tf_debug

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

_HASH_BUCKET_SIZE = 1000

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}


def _download_and_clean_file(filename, url):
  """Downloads data from url, and makes changes to match the CSV format."""
  temp_file, _ = urllib.request.urlretrieve(url)
  with tf.gfile.Open(temp_file, 'r') as temp_eval_file:
    with tf.gfile.Open(filename, 'w') as eval_file:
      for line in temp_eval_file:
        line = line.strip()
        line = line.replace(', ', ',')
        if not line or ',' not in line:
          continue
        if line[-1] == '.':
          line = line[:-1]
        line += '\n'
        eval_file.write(line)
  tf.gfile.Remove(temp_file)


def download(data_dir):
  """Download census data if it is not already present."""
  tf.gfile.MakeDirs(data_dir)

  training_file_path = os.path.join(data_dir, TRAINING_FILE)
  if not tf.gfile.Exists(training_file_path):
    _download_and_clean_file(training_file_path, TRAINING_URL)

  eval_file_path = os.path.join(data_dir, EVAL_FILE)
  if not tf.gfile.Exists(eval_file_path):
    _download_and_clean_file(eval_file_path, EVAL_URL)


def build_model_columns():
  """Builds a set of wide and deep feature columns."""
  # Continuous variable columns
  age = tf.feature_column.numeric_column('age')
  education_num = tf.feature_column.numeric_column('education_num')
  capital_gain = tf.feature_column.numeric_column('capital_gain')
  capital_loss = tf.feature_column.numeric_column('capital_loss')
  hours_per_week = tf.feature_column.numeric_column('hours_per_week')

  education = tf.feature_column.categorical_column_with_vocabulary_list(
      'education', [
          'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
          'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
          '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

  marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
      'marital_status', [
          'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
          'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

  relationship = tf.feature_column.categorical_column_with_vocabulary_list(
      'relationship', [
          'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
          'Other-relative'])

  workclass = tf.feature_column.categorical_column_with_vocabulary_list(
      'workclass', [
          'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
          'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

  # To show an example of hashing:
  occupation = tf.feature_column.categorical_column_with_hash_bucket(
      'occupation', hash_bucket_size=_HASH_BUCKET_SIZE)

  # Transformations.
  age_buckets = tf.feature_column.bucketized_column(
      age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

  # Wide columns and deep columns.
  base_columns = [
      education, marital_status, relationship, workclass, occupation,
      age_buckets,
  ]

  # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/crossed_column
  crossed_columns = [
      tf.feature_column.crossed_column(
          ['education', 'occupation'], hash_bucket_size=_HASH_BUCKET_SIZE),
      tf.feature_column.crossed_column(
          [age_buckets, 'education', 'occupation'],
          hash_bucket_size=_HASH_BUCKET_SIZE),
  ]

  wide_columns = base_columns + crossed_columns

  # https://www.tensorflow.org/api_docs/python/tf/feature_column/indicator_column
  deep_columns = [
      age,
      education_num,
      capital_gain,
      capital_loss,
      hours_per_week,
      tf.feature_column.indicator_column(workclass),
      tf.feature_column.indicator_column(education),
      tf.feature_column.indicator_column(marital_status),
      tf.feature_column.indicator_column(relationship),
      # To show an example of embedding
      # https://www.tensorflow.org/api_docs/python/tf/feature_column/embedding_column
      tf.feature_column.embedding_column(occupation, dimension=8),
  ]

  return wide_columns, deep_columns


def build_estimator(model_dir, model_type, machine_type):
  """Build an estimator appropriate for the given model type."""
  wide_columns, deep_columns = build_model_columns()
  hidden_units = [100, 75, 50, 25]

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.

  if machine_type == "CPU":
      run_config = tf.estimator.RunConfig().replace(
          session_config=tf.ConfigProto(device_count={'GPU': 0}))
  else:
      run_config = tf.estimator.RunConfig().replace(
          session_config=tf.ConfigProto(device_count={'GPU': 2}))

  if model_type == 'wide':
    return tf.estimator.LinearClassifier(
        model_dir=model_dir,
        feature_columns=wide_columns,
        config=run_config)
  elif model_type == 'deep':
    return tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config)
  else:
    # https://www.tensorflow.org/api_docs/python/tf/estimator/DNNLinearCombinedClassifier
    return tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config)


def input_fn(data_file, num_epochs, shuffle, batch_size):
  """Generate an input function for the Estimator."""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have run census_dataset.py and '
      'set the --data_dir argument to the correct path.' % data_file)

  def parse_csv(value):
    tf.logging.info('Parsing {}'.format(data_file))
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('income_bracket')
    classes = tf.equal(labels, '>50K')  # binary classification
    return features, classes

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

  dataset = dataset.map(parse_csv, num_parallel_calls=5)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  return dataset

def pred_input_fn(data_file, num_epochs, shuffle, batch_size):
    assert tf.gfile.Exists(data_file), (
        '%s not found.Please make sure you have run data_download.py and '
        'set the --data_dir argument to the correct path.' % data_file)

    def parse_csv(value):
        print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        return features

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)


    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    # dataset = dataset.batch(batch_size)
    return dataset


def exec_pred(flags, model, pred_input_fn):
    # model.predict(
    #     input_fn=pred_input_fn
    # )
    results = model.predict(input_fn=pred_input_fn)
    for i, key in enumerate(results):
        print(i, key)
        if i == 100:
            break



def define_data_download_flags():
  """Add flags specifying data download arguments."""
  flags.DEFINE_string(
      name="data_dir", default="/tmp/census_data/",
      help=flags_core.help_wrap(
          "Directory to download and extract data."))



  if flags.execute_type == "train":
      # Clean up the model directory if present
      shutil.rmtree(flags.model_dir, ignore_errors=True)

  model = build_estimator(flags.model_dir, flags.model_type, flags.machine_type)

def main(_):
  download(flags.FLAGS.data_dir)
  train_file = os.path.join(flags.data_dir, 'adult.data')
  test_file = os.path.join(flags.data_dir, 'adult.test')

  # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
  def train_input_fn():
    return input_fn(train_file, flags.epochs_per_eval, True, flags.batch_size)

  def eval_input_fn():
    return input_fn(test_file, 1, False, flags.batch_size)

  def pred_input_fn():
      return input_fn(test_file, 1, False, 1)

  # Graphs
  train_hooks = hooks_helper.get_train_hooks(
      flags.hooks, batch_size=flags.batch_size,
      tensors_to_log={'average_loss': 'head/truediv',
                      'loss': 'head/weighted_loss/Sum'})

  if flags.execute_type == "test":

      exec_pred(flags, model, pred_input_fn)

      exit(0)

  # tensorboard debugger
  # train_hooks = [tf_debug.TensorBoardDebugHook("localhost:6005")]

  # Train and evaluate the model every `FLAGS.epochs_between_evals` epochs.
  for n in range(flags.train_epochs // flags.epochs_between_evals):
    model.train(input_fn=train_input_fn, hooks=train_hooks)
    results = model.evaluate(input_fn=eval_input_fn)

    # Display evaluation metrics
    print('Results at epoch', (n + 1) * flags.epochs_between_evals)
    print('-' * 60)

    for key in sorted(results):
      print('%s: %s' % (key, results[key]))


class WideDeepArgParser(argparse.ArgumentParser):
  """Argument parser for running the wide deep model."""

  def __init__(self):
    super(WideDeepArgParser, self).__init__(parents=[parsers.BaseParser()])
    self.add_argument(
        '--model_type', '-mt', type=str, default='wide_deep',
        choices=['wide', 'deep', 'wide_deep'],
        help='[default %(default)s] Valid model types: wide, deep, wide_deep.',
        metavar='<MT>')
    self.add_argument(
        '--execute_type', '-et', type=str, default='train',
        choices=['train', 'test'],
        help='[default %(default)s] model execution types: train, test.')
    self.add_argument(
        '--machine_type', '-mct', type=str, default='GPU',
        choices=['CPU', 'GPU'],
        help='[default %(default)s] machine execution types: CPU, GPU.')
    self.set_defaults(
        data_dir='./census_data',
        model_dir='./census_model',
        train_epochs=40,
        epochs_between_evals=2,
        batch_size=40,
        epochs_per_eval=2)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_data_download_flags()
  absl_app.run(main)
