import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import ADASYN

import rdkit.Chem as Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import MACCSkeys
import os

# accessing the datasets train and test from the datasets folder, and reading them as csv using pandas
train = pd.read_csv('datasets/train.csv')
test = pd.read_csv('datasets/test.csv')

## This function calculates the molcular descriptors of each SMILE and appends to the descriptors array.
## Input: a string contaning SMILE notation of the chemical structure, array of all the descriptor calculators
## Output: an array with calculated descriptors of the SMILE chemical structure
def calculate_descriptors(smiles_string, descriptor_calculators):
  mol = Chem.MolFromSmiles(smiles_string)
  descriptors = []
  for descriptor_calculator in descriptor_calculators:
    if mol is not None:
      value = descriptor_calculator.CalcDescriptors(mol)
    else:
      value = [np.zeros(208)]
    descriptors.append(value)
  return descriptors

# This function is for preprocessing of the dataset. First it checks if the input is test dataset or not, and then takes the id as 'x' if it is test
# and as 'Id' if it is train dataset. As XGBoost only takes 0's and 1's as target variable, I am converting 2's and 1's if it is train dataset (the test 
# dataset does not consist of target varaibles). We split the id column into 'SMILE' and 'AID' columns and then converted the AID columns into integer type.
# After filling all the none values with 0, we extract the molecular descriptors of the SMILE and save as pdf. Then calculate MACCS features and add them in
# chunks to reduce data loss. Combine the molecular descriptors data and MAACS features data to produce final dataset. 
# Input: a dataframe consisting of train or test dataset, a boolean consisting if it is test or train dataset.
# Output: a dataframe consisting of the final dataset.
def featureExtraction(dataset, isTest):
  id = ''
  if isTest:
    id = 'x'
  else:
    dataset['Expected'].replace({1: 0, 2: 1}, inplace=True)
    id = 'Id'
  
  dataset[['SMILE', 'AID']] = dataset[id].str.split(';', expand=True)
  dataset["AID"] = dataset["AID"].astype(int)
  dataset = dataset.fillna(0)

  smiles = dataset['SMILE']

  print('Adding molecular descriptors features')
  descriptor_names = [desc_name[0] for desc_name in Descriptors.descList]
  descriptor_calculators = [MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)]
  
  descriptor_values = [calculate_descriptors(smile, descriptor_calculators) for smile in smiles]
  descriptor_df = pd.DataFrame(descriptor_values, columns=descriptor_names, index=dataset.index)
  dataset[descriptor_names] = descriptor_df

  # dataset[descriptor_names] = pd.DataFrame([calculate_descriptors(smile, descriptor_calculators) for smile in smiles], index=dataset.index)

  if isTest:
    dataset.to_csv('datasets/molecular_descriptors_test.csv', index = False)
  else:
    print("Saving MD")
    dataset.to_csv('datasets/molecular_descriptors_train.csv', index = False)

  print('Adding MACCS keys features')  
  MaccsKeysDataframe = pd.DataFrame(columns=[f'maccs_{i+1}' for i in range(167)])
  chunk_size = 5000
  print('Adding other features')
  for i in range(0, len(smiles), chunk_size):
    chunk_smiles = smiles[i:i+chunk_size]
    maccs_keys_list = []
    for smile in chunk_smiles:
      if Chem.MolFromSmiles(smile) is not None:
        mol = Chem.MolFromSmiles(smile)
        maccs_keys = MACCSkeys.GenMACCSKeys(mol)
        keys = [0 for i in range(167)]
        i = 0
        for maccs in maccs_keys:
          col = 'maccs_' + str(i)
          keys[i] = maccs
          i = i + 1
        maccs_keys_list.append(keys)
      else:
        maccs_keys_list.append(np.zeros(167))
    maccs_df = pd.DataFrame(data=maccs_keys_list, columns=[f'maccs_{i+1}' for i in range(167)])
    MaccsKeysDataframe = pd.concat([MaccsKeysDataframe, maccs_df], axis=0, ignore_index=True)

  if isTest:
    MaccsKeysDataframe.to_csv('datasets/maccskeys_test.csv', index = False)
  else:
    MaccsKeysDataframe.to_csv('datasets/maccskeys_train.csv', index = False)

  finalDataset = pd.concat([dataset, MaccsKeysDataframe], axis = 1)
  if not isTest:
    finalDataset = finalDataset.dropna()
    finalDataset.to_csv('datasets/processed_data_train.csv', index = False)
  else:
    finalDataset.to_csv('datasets/processed_data_train.csv', index = False)

  return finalDataset

# get train and test datasets from the location
train_location = 'datasets/processed_data_train.csv'
test_location = 'datasets/processed_data_test.csv'

# If the train dataset is not present in the train location, then extract features, else read the csv file as a dataframe using pandas.
if not os.path.isfile(train_location):
  print('No processed training data')
  dataset = featureExtraction(train, False)
else:
  dataset = pd.read_csv(train_location)

# If the test dataset is not present in the test location, then extract features, else read the csv file as a dataframe using pandas.
if not os.path.isfile(test_location):
  print('No processed testing data')
  test = featureExtraction(test, True)
else:
  test = pd.read_csv(test_location)

# y is the target variable consisting of the target values. train['Expected'] is the target value.
y = dataset['Expected']
# Dropping unnecessary values from the dataset
X = dataset.drop(['Expected', 'Id', 'SMILE'], axis = 1)
# Filling the none column values with 0 in train dataset
X = X.fillna(0)
# Convert all the 1s and 2s to 0s and 1s ( for XGBoost )
y.replace({1: 0, 2: 1}, inplace=True)
# Filling the none column values with 0 in target variable
y = y.fillna(0)
# Filling the none column values with 0 in test dataset
test = test.fillna(0)

# To balance the data, Adaptive Synthetic algorithm has been used with oversampling 25%. 
adasyn = ADASYN(random_state=42, sampling_strategy = 0.25)
X_resampled, y_resampled = adasyn.fit_resample(X, y)

print("Class counts after ADASYN oversampling:")
# printing number of 0s and 1s in the target variable
class_counts2 = y_resampled.value_counts()
print(class_counts2)

# This method extracts highly correlated features from a given dataframe.
# Input: a dataframe consisting of test or train dataset, a float value consisting of the threshold value
# Output: an array consisting of the column numbers that are highly correlated features from the given dataframe.
def correlation_features(X, corr_threshold):
  col_corr = set()
  matrix = X.corr()
  for i in range(len(matrix.columns)):
    for j in range(i):
      if abs(matrix.iloc[i, j]) > corr_threshold:
        col_corr.add(matrix.columns[i])
  return col_corr   

# Taking a copy of the resampled data
temp_X = X_resampled
# Calculating correlation features with 0.7 as threshold
corr_feature = correlation_features(temp_X, 0.7)

# Dropping the rest of the columns
X_resampled=X_resampled.drop(columns=[col for col in X_resampled.columns if col not in corr_feature],axis=1)
test=test.drop(columns=[col for col in test.columns if col not in corr_feature],axis=1)

# Initializing parameters for the XGBoost model for RandomizedSearchCV.
xgb_params = {
    'learning_rate': [0.1, 0.5, 0.01],
    'max_depth': [3, 5, 7],
    'n_estimators': [300],
    'subsample': [0.5, 0.8, 1.0],
    'colsample_bytree': [0.5, 0.8, 1.0],
    'gamma': [0, 0.1, 0.5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 0.1, 1],
    'min_child_weight': [1, 3, 5]
}

# Performs RandomizedSearchCV with XGBoost classifier to find the best parameters
# Input: an array consisting of train dataset, an array consisting of test dataset
def GridSearchXGB(X_train, y_train):
  xgbClassifier = XGBClassifier()
  xgb_grid = RandomizedSearchCV(xgbClassifier, xgb_params, cv=5, n_jobs=-1)
  xgb_grid.fit(X_train, y_train)
  xgb_best_params = xgb_grid.best_params_
  xgb_best_score = xgb_grid.best_score_
  print('XGB Best parameters:', xgb_best_params)
  print('XGB Best score:', xgb_best_score)

# Performs K-fold cross validation with XGBoost Classifier
def xgbKFold(X_train, X_test, y_train, y_test, counter):
  f1 = []
  #define the model with best parameters
  model = XGBClassifier(random_state=42, colsample_bytree= 1.0, gamma= 0, learning_rate= 0.5, max_depth= 5, 
                        min_child_weight= 5, n_estimators= 400, reg_alpha= 0, reg_lambda= 0.1, subsample= 0.8)
  for i in range(counter):
    print('K-Fold: ',i)

    model.fit(X_train[i], y_train[i])
    y_pred = model.predict(X_test[i])
    f1.append(f1_score(y_test[i],y_pred, zero_division=1))
  return model, f1

# Prepare the dataset for k-folds.
# Input: dataframes consisting of train dataset and test dataset
# Returns: 4 variable consisting of X_train, X_test, y_train and y_test with number of k-fold
def prepareDatasetKFold(X, y):

  X = X.to_numpy()
  np.set_printoptions(precision=2, suppress=True)
  # Applying K-Fold cross validation
  kf = KFold(n_splits=4)
  kf.get_n_splits(X)
  #Initializing X_train, X_test, y_train, y_test as empty arrays 
  X_train, X_test, y_train, y_test = [], [], [], []

  # For each train and test index in split of X, y : Append X[train_index], y[train_index] to train dataset and X[test_index], y[test_index] to test dataset
  for train_index, test_index in kf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train.append(X[train_index])
    X_test.append(X[test_index])
    y_train.append(y[train_index])
    y_test.append(y[test_index])
  # Return the test and train dataset with k-fold (currently 4)
  return X_train, X_test, y_train, y_test, 4


# Prepare the K-Fold dataset
X_train, X_test, y_train, y_test, counter = prepareDatasetKFold(X_resampled, y_resampled)

# Train the model
XGBmodel, XGBf1 = xgbKFold(X_train, X_test, y_train, y_test, counter)
print("XGBoost f1: ", np.mean(XGBf1))

# Predict on the test dataset
test_pred_xgbK = XGBmodel.predict(test)
# Convert the 0s and 1s to 1s and 2s
converted_test_xgbK = [1 if x == 0 else 2 if x == 1 else x for x in test_pred_xgbK]
# Add the data to the dataframe
test_predictions = pd.DataFrame({'Predicted': converted_test_xgbK})
test_predictions["Predicted"] = test_predictions["Predicted"].apply(int)

# Extract the 'x' of the dataset from test.csv and add it as index
original_test = pd.read_csv('datasets/test.csv')
test_predictions.index = original_test['x']

# Save the predicted file
test_predictions.to_csv('output/test_prediction_xgbK.csv', index_label='Id')

print("Saved the predicted file successfully.")
