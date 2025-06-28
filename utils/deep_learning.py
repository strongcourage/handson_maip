import json
import os
import shutil
from pathlib import Path

from trafficToFeature import trafficToFeatures
from createDatasetMMT import createTrainTestSet
from trainer import train_model

def prepare_training_location(buildId):
  # All outputs go under ./<buildId>/
  output_base_path = os.path.join(str(Path.cwd()), buildId)
  print('Prepare training locations: ', output_base_path)
  # Prepare the location
  if not os.path.exists(output_base_path):
    os.makedirs(output_base_path)
  # Traffic to Feature
  output_pickle_files_location = os.path.join(output_base_path, 'pickles/')
  if not os.path.exists(output_pickle_files_location):
    os.makedirs(output_pickle_files_location)
  # Create dataset
  output_datasets_location = os.path.join(output_base_path, 'datasets/')
  if not os.path.exists(output_datasets_location):
    os.makedirs(output_datasets_location)
  # Training model
  output_result_location = os.path.join(output_base_path, 'results/')
  if not os.path.exists(output_result_location):
    os.makedirs(output_result_location)
  return output_pickle_files_location, output_datasets_location, output_result_location, output_base_path

def train_model_from_csv(datasets, training_ratio, buildId, training_parameters = {
  "nb_epoch_cnn": 5,
  "batch_size_cnn": 16,
  "nb_epoch_sae": 2,
  "batch_size_sae": 32
}):
  """A completed flow of training model from datasets

  Args:
    datasets (Array): List of dataset to be used for training
    training_ratio (Number): the ratio of training dataset and testing dataset
    training_parameters (Object): Parameter for training
  """
  print('Build id: ' + buildId)
  # Prepare locations
  output_pickle_files_location, output_datasets_location, output_result_location, output_base_path = prepare_training_location(buildId)
  # return trainId
  # Traffic to Feature
  for dtset in datasets:
    csvPath = dtset['csvPath']
    isAttack = dtset['isAttack']
    pickle_file_path = os.path.join(output_pickle_files_location, os.path.basename(csvPath).split('/')[-1] + '.pkl')
    trafficToFeatures(csvPath, pickle_file_path, isAttack)
    print('A new pickle file at: ' + pickle_file_path + '(' + str(isAttack) + ')')

  # Create dataset
  createTrainTestSet(output_pickle_files_location, training_ratio, output_datasets_location)
  print('Datasets have been created at: ' + output_datasets_location)
  train_data_path = os.path.join(output_datasets_location, 'Train_samples.csv')
  test_data_path = os.path.join(output_datasets_location, 'Test_samples.csv')
  # training model
  train_model(train_data_path, test_data_path, output_result_location, training_parameters['nb_epoch_cnn'], training_parameters['nb_epoch_sae'], training_parameters['batch_size_cnn'], training_parameters['batch_size_sae'])
  print('New model has been created at: ' + output_result_location)
  # Save model as <buildId>/<buildId>.h5
  model_path = os.path.join(output_base_path, buildId + '.h5')
  shutil.copy(os.path.join(output_result_location, 'model.h5'), model_path)
  print(f'Model saved at: {model_path}')
  return buildId

def start_training_model(buildId, path_to_training_config_file):
  if not os.path.exists(path_to_training_config_file):
    print("ERROR: Training config file does not exist: " + path_to_training_config_file)
    return ''
  else:
    print('Reading the training configuration file')
    f = open(path_to_training_config_file)
    trainingConfig = json.load(f)
    datasets = trainingConfig['datasets']
    training_ratio = trainingConfig['training_ratio']
    training_parameters = trainingConfig['training_parameters']
    return train_model_from_csv(datasets, training_ratio, buildId, training_parameters)

if __name__ == "__main__":
  import sys
  print(sys.argv)
  if len(sys.argv) != 3:
    print('Invalid inputs')
    print('python deep_learning.py buildId path_to_training_config_file.json')
  else:
    start_training_model(sys.argv[1], sys.argv[2])