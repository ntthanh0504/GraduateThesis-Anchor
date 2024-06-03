import copy
import sklearn
import numpy as np
import lime
import os
import pandas as pd

# from __future__ import print_function
import lime.lime_tabular
# import string


class Bunch(object):
    """Container object for datasets: dictionary-like object that"""

    def __init__(self, dict):
        self.__dict__.update(dict)

def load_dataset(dataset_name, balance=False, discretize=True, dataset_folder='./'):
    if dataset_name == 'adult':
        feature_names = ["Age", "Workclass", "fnlwgt", "Education",
                         "Education-Num", "Marital Status", "Occupation",
                         "Relationship", "Race", "Sex", "Capital Gain",
                         "Capital Loss", "Hours per week", "Country", 'Income']
        used_feature_indices = [0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        categorical_feature_indices = [1, 3, 5, 6, 7, 8, 9, 10, 11, 13]
        education_mapping = {
            '10th': 'Dropout', '11th': 'Dropout', '12th': 'Dropout', '1st-4th':
            'Dropout', '5th-6th': 'Dropout', '7th-8th': 'Dropout', '9th':
            'Dropout', 'Preschool': 'Dropout', 'HS-grad': 'High School grad',
            'Some-college': 'High School grad', 'Masters': 'Masters',
            'Prof-school': 'Prof-School', 'Assoc-acdm': 'Associates',
            'Assoc-voc': 'Associates',
        }
        occupation_mapping = {
            "Adm-clerical": "Admin", "Armed-Forces": "Military",
            "Craft-repair": "Blue-Collar", "Exec-managerial": "White-Collar",
            "Farming-fishing": "Blue-Collar", "Handlers-cleaners":
            "Blue-Collar", "Machine-op-inspct": "Blue-Collar", "Other-service":
            "Service", "Priv-house-serv": "Service", "Prof-specialty":
            "Professional", "Protective-serv": "Other", "Sales":
            "Sales", "Tech-support": "Other", "Transport-moving":
            "Blue-Collar",
        }
        country_mapping = {
            'Cambodia': 'SE-Asia', 'Canada': 'British-Commonwealth', 'China':
            'China', 'Columbia': 'South-America', 'Cuba': 'Other',
            'Dominican-Republic': 'Latin-America', 'Ecuador': 'South-America',
            'El-Salvador': 'South-America', 'England': 'British-Commonwealth',
            'France': 'Euro_1', 'Germany': 'Euro_1', 'Greece': 'Euro_2',
            'Guatemala': 'Latin-America', 'Haiti': 'Latin-America',
            'Holand-Netherlands': 'Euro_1', 'Honduras': 'Latin-America',
            'Hong': 'China', 'Hungary': 'Euro_2', 'India':
            'British-Commonwealth', 'Iran': 'Other', 'Ireland':
            'British-Commonwealth', 'Italy': 'Euro_1', 'Jamaica':
            'Latin-America', 'Japan': 'Other', 'Laos': 'SE-Asia', 'Mexico':
            'Latin-America', 'Nicaragua': 'Latin-America',
            'Outlying-US(Guam-USVI-etc)': 'Latin-America', 'Peru':
            'South-America', 'Philippines': 'SE-Asia', 'Poland': 'Euro_2',
            'Portugal': 'Euro_2', 'Puerto-Rico': 'Latin-America', 'Scotland':
            'British-Commonwealth', 'South': 'Euro_2', 'Taiwan': 'China',
            'Thailand': 'SE-Asia', 'Trinadad&Tobago': 'Latin-America',
            'United-States': 'United-States', 'Vietnam': 'SE-Asia'
        }
        married_mapping = {
            'Never-married': 'Never-Married', 'Married-AF-spouse': 'Married',
            'Married-civ-spouse': 'Married', 'Married-spouse-absent':
            'Separated', 'Separated': 'Separated', 'Divorced':
            'Separated', 'Widowed': 'Widowed'
        }
        label_mapping = {'<=50K': 'Less than $50,000',
                         '>50K': 'More than $50,000'}
        
        def categorize_cap_gains(x):
            x = x.astype(float)
            bins = [0, np.median(x[x > 0]), float('inf')]
            categories = ['None', 'Low', 'High']
            digitized = np.digitize(x, bins, right=True).astype('|S128')
            return map_array_values(digitized, dict(enumerate(categories)))
        
        def map_array_values(array, value_map):
            """
            Args:
                array: numpy array
                value_map: dictionary, { src : target }
            Returns:
                ret: numpy array
            Purpose:
                Replace the values in the array with the values in the value_map
            """

            ret = array.copy()
            for src, target in value_map.items():
                ret[ret == src] = target
            return ret

        transformations = {
            3: lambda x: map_array_values(x, education_mapping),
            5: lambda x: map_array_values(x, married_mapping),
            6: lambda x: map_array_values(x, occupation_mapping),
            10: categorize_cap_gains,
            11: categorize_cap_gains,
            13: lambda x: map_array_values(x, country_mapping),
            14: lambda x: map_array_values(x, label_mapping),
        }
        
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'adult/adult.data'), -1, ', ',
            feature_names=feature_names, used_feature_indices=used_feature_indices,
            categorical_feature_indices=categorical_feature_indices, discretize=discretize,
            balance=balance, feature_transformations=transformations)
        
    elif dataset_name == 'diabetes':
        categorical_feature_indices = [2, 3, 4, 5, 6, 7, 8, 10, 11, 18, 19, 20, 22,
                                       23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                                       35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                                       47, 48]
        label_mapping = {'<30': 'YES', '>30': 'YES'}
        transformations = {
            49: lambda x: map_array_values(x, label_mapping),
        }
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'diabetes/diabetic_data.csv'), -1, ',',
            used_feature_indices=range(2, 49),
            categorical_feature_indices=categorical_feature_indices, discretize=discretize,
            balance=balance, feature_transformations=transformations)
        
    elif dataset_name == 'default':
        categorical_feature_indices = [2, 3, 4, 6, 7, 8, 9, 10, 11]
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'default/default.csv'), -1, ',',
            used_feature_indices=range(1, 24),
            categorical_feature_indices=categorical_feature_indices, discretize=discretize,
            balance=balance)
        
    elif dataset_name == 'recidivism':
        used_feature_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        feature_names = ['Race', 'Alcohol', 'Junky', 'Supervised Release',
                         'Married', 'Felony', 'WorkRelease',
                         'Crime against Property', 'Crime against Person',
                         'Gender', 'Priors', 'YearsSchool', 'PrisonViolations',
                         'Age', 'MonthsServed', '', 'Recidivism']

        def map_violations(x):
            """
            Args: x: numpy array
            Returns: ret: numpy array
            Purpose: Map the number of violations to a string
            """
            x = x.astype(float)
            d = np.digitize(x, [0, 5, float('inf')], right=True).astype('|S128')
            return map_array_values(d, {'0': 'NO', '1': '1 to 5', '2': 'More than 5'})

        def map_priors(x):
            """
            Args: x: numpy array
            Returns: ret: numpy array
            Purpose: Map the number of priors to a string
            """
            x = x.astype(float)
            d = np.digitize(x, [-1, 0, 5, float('inf')], right=True).astype('|S128')
            return map_array_values(d, {'0': 'UNKNOWN', '1': 'NO', '2': '1 to 5', '3': 'More than 5'})
        
        def replace_binary_values(array, values):
            """
            Args:
                array: numpy array
                values: list of strings
            Returns:
                ret: numpy array
            Purpose:
                Replace the binary values in the array with the values in the values list
            """
            return map_array_values(array, {'0': values[0], '1': values[1]})
        transformations = {
            0: lambda x: replace_binary_values(x, ['Black', 'White']),
            1: lambda x: replace_binary_values(x, ['No', 'Yes']),
            2: lambda x: replace_binary_values(x, ['No', 'Yes']),
            3: lambda x: replace_binary_values(x, ['No', 'Yes']),
            4: lambda x: replace_binary_values(x, ['No', 'Married']),
            5: lambda x: replace_binary_values(x, ['No', 'Yes']),
            6: lambda x: replace_binary_values(x, ['No', 'Yes']),
            7: lambda x: replace_binary_values(x, ['No', 'Yes']),
            8: lambda x: replace_binary_values(x, ['No', 'Yes']),
            9: lambda x: replace_binary_values(x, ['Female', 'Male']),
            10: lambda x: map_priors(x),
            12: lambda x: map_violations(x),
            13: lambda x: (x.astype(float) / 12).astype(int),
            16: lambda x: replace_binary_values(x, ['No more crimes', 'Re-arrested'])
        }

        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'recidivism/Data_1980.csv'), 16,
            feature_names=feature_names, discretize=discretize,
            used_feature_indices=used_feature_indices, balance=balance,
            feature_transformations=transformations, skip_first=True)
        
    elif dataset_name == 'lending':
        def filter_fn(data):
            to_remove = ['Does not meet the credit policy. Status:Charged Off',
                         'Does not meet the credit policy. Status:Fully Paid',
                         'In Grace Period', '-999', 'Current']
            for x in to_remove:
                data = data[data[:, 16] != x]
            return data
        
        bad_statuses = set(
            ["Late (16-30 days)", "Late (31-120 days)", "Default", "Charged Off"])
        transformations = {
            16: lambda x: np.array([y in bad_statuses for y in x]).astype(int),
            19: lambda x: np.array([len(y) for y in x]).astype(int),
            6: lambda x: np.array([y.strip('%') if y else -1 for y in x]).astype(float),
            35: lambda x: np.array([y.strip('%') if y else -1 for y in x]).astype(float),
        }
        used_feature_indices = [2, 12, 13, 19, 29, 35, 51, 52, 109]
        categorical_feature_indices = [12, 109]
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'lendingclub/LoanStats3a_securev1.csv'), 16, ',', 
            used_feature_indices=used_feature_indices,
            feature_transformations=transformations, fill_na='-999',
            categorical_feature_indices=categorical_feature_indices, discretize=discretize,
            filter_fn=filter_fn, balance=True)
        
        dataset.class_names = ['Good Loan', 'Bad Loan']

    return dataset

def load_csv_dataset(data, target_idx, delimiter=',',
                     feature_names=None, categorical_feature_indices=None,
                     used_feature_indices=None, feature_transformations=None,
                     discretize=False, balance=False, fill_na='-1', filter_fn=None, skip_first=False):
    """
    Args: 
        data: string or file. String is a path to a csv file. target_idx: int
        feature_names: list of strings, names for the features, if None, first row is used
        categorical_feature_indices: list of ints, indices of categorical features, if None, we infer from the data
        used_feature_indices: list of ints, indices of features to use, if None all are used, except target_idx
        discretize: boolean, whether to discretize the data
        balance: boolean, whether to balance the data
        fill_na: string, what to fill nans with
        filter_fn: function, takes data and returns a subset of it
        skip_first: boolean, whether to skip the first row

    Returns:
        data: Bunch

    Purpose:
        Load a dataset from a csv file, with the option to preprocess it.
    """
    ret = Bunch({})
    encoder = sklearn.preprocessing.LabelEncoder()

    # Get dataset from given path
    try:
        data = np.genfromtxt(data, delimiter=delimiter, dtype='|S128')
    except:
        data = pd.read_csv(data, header=None, delimiter=delimiter,
                           na_filter=True, dtype=str).fillna(fill_na).values

    # If no feature transformations are given, set it to an empty dictionary
    if feature_transformations is None:
        feature_transformations = {}

    # If target index is negative, set it to the last index
    if target_idx < 0:
        target_idx = data.shape[1] + target_idx

    # If no feature names are given, set it to the first row of the data
    if feature_names is None:
        feature_names = list(data[0])
        data = data[1:]
    else:
        feature_names = copy.deepcopy(feature_names)

    # If we are skipping the first row, remove it from the data
    if skip_first:
        data = data[1:]

    # If no categorical features are given, infer them from the data
    if filter_fn is not None:
        data = filter_fn(data)

    # Transform the data using the given feature transformations
    for feature, func in feature_transformations.items():
        data[:, feature] = func(data[:, feature])

    # Encode the target labels and add some attributes to the dataset
    labels = data[:, target_idx]
    ret.labels = encoder.fit_transform(labels)
    labels = ret.labels
    ret.class_names = list(encoder.classes_)
    ret.class_target = feature_names[target_idx]

    # If features to use are given, use them, otherwise use all features
    if used_feature_indices is not None:
        data = data[:, used_feature_indices]
        feature_names = [feature_names[i] for i in used_feature_indices]
        # Update the categorical features to use
        if categorical_feature_indices is not None:
            categorical_feature_indices = [used_feature_indices.index(
                x) for x in categorical_feature_indices]
    else:
        data = np.delete(data, target_idx, 1)
        feature_names.pop(target_idx)
        if categorical_feature_indices:
            categorical_feature_indices = [
                x if x < target_idx else x - 1 for x in categorical_feature_indices]

    # If no categorical features are given, infer them from the data
    if categorical_feature_indices is None:
        categorical_feature_indices = [f for f in range(
            data.shape[1]) if len(np.unique(data[:, f])) < 20]

    # Encode the categorical features
    categorical_names = {}
    for feature_index in categorical_feature_indices:
        encoder.fit(data[:, feature_index])
        data[:, feature_index] = encoder.transform(data[:, feature_index])
        categorical_names[feature_index] = encoder.classes_

    # Convert the data to float
    data = data.astype(float)

    # Discretize the data by quartiles
    # It means that each feature is divided into 4 bins of equal frequency, and marked as 0, 1, 2, 3
    ordinal_features = []
    if discretize:
        disc = lime.lime_tabular.QuartileDiscretizer(data,
                                                     categorical_feature_indices,
                                                     feature_names)
        data = disc.discretize(data)
        ordinal_features = [x for x in range(data.shape[1])
                            if x not in categorical_feature_indices]
        categorical_feature_indices = list(range(data.shape[1]))
        categorical_names.update(disc.names)

    # Decode the categorical features to their original values
    for x in categorical_names:
        categorical_names[x] = [y.decode() if type(
            y) == np.bytes_ else y for y in categorical_names[x]]

    # Add the data and some other attributes to the dataset
    ret.ordinal_features = ordinal_features
    ret.categorical_feature_indices = categorical_feature_indices
    ret.categorical_names = categorical_names
    ret.feature_names = feature_names

    np.random.seed(1)

    # Improve the balance of the dataset
    # It means that the number of samples in each class is equal
    if balance:
        min_labels = np.min(np.bincount(labels))
        idxs = np.concatenate([np.random.choice(np.where(labels == label)[0], min_labels) 
                               for label in np.unique(labels)])
        data = data[idxs]
        labels = labels[idxs]
        ret.data = data
        ret.labels = labels

    # Add the data to the dataset
    ret.data = data

    # Split the data into train and test sets
    splits = sklearn.model_selection.ShuffleSplit(
        n_splits=1, test_size=.2, random_state=1)
    train_idx, test_idx = next(splits.split(data))
    ret.train = data[train_idx]
    ret.labels_train = ret.labels[train_idx]

    # Split the test set into validation and test sets
    cv_splits = sklearn.model_selection.ShuffleSplit(
        n_splits=1, test_size=.5, random_state=1)
    cv_idx, test_idx = next(cv_splits.split(test_idx))
    ret.validation = data[cv_idx]
    ret.labels_validation = ret.labels[cv_idx]
    ret.test = data[test_idx]
    ret.labels_test = ret.labels[test_idx]
    ret.test_idx = test_idx
    ret.validation_idx = cv_idx
    ret.train_idx = train_idx

    return ret


# if __name__ == '__main__':
#     # make sure you have adult/adult.data inside dataset_folder
#     dataset_folder = 'F:/References/anchor-master/anchor-master/anchor-experiments-master/datasets/'
#     dataset = load_dataset('adult', balance=True,
#                            dataset_folder=dataset_folder, discretize=True)
