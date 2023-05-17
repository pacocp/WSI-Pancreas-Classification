import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GroupKFold

def patient_split_full(dataset, random_state=0):
    """Perform patient split of any of the previously defined datasets.
    """
    patients_unique = np.unique(dataset.Patient_ID)
    patients_train, patients_test = train_test_split(
        patients_unique, test_size=0.2, random_state=random_state)

    indices = np.arange(len(dataset))
    train_idx = indices[np.any(np.array(dataset.Patient_ID)[:, np.newaxis] == 
                        np.array(patients_train)[np.newaxis], axis=1)]
    test_idx = indices[np.any(np.array(dataset.Patient_ID)[:, np.newaxis] == 
                        np.array(patients_test)[np.newaxis], axis=1)]

    return train_idx, test_idx

def patient_split(dataset, random_state=0):
    """Perform patient split of any of the previously defined datasets.
    """
    patients_unique = np.unique(dataset.Patient_ID)
    patients_train, patients_test = train_test_split(
        patients_unique, test_size=0.2, random_state=random_state)
    patients_train, patients_val = train_test_split(
        patients_train, test_size=0.2, random_state=random_state)

    indices = np.arange(len(dataset))
    train_idx = indices[np.any(np.array(dataset.Patient_ID)[:, np.newaxis] == 
                        np.array(patients_train)[np.newaxis], axis=1)]
    valid_idx = indices[np.any(np.array(dataset.Patient_ID)[:, np.newaxis] == 
                        np.array(patients_val)[np.newaxis], axis=1)]
    test_idx = indices[np.any(np.array(dataset.Patient_ID)[:, np.newaxis] == 
                        np.array(patients_test)[np.newaxis], axis=1)]

    return train_idx, valid_idx, test_idx

#def patient_group_k_fold(dataset, k, random_state=0):
#    group_kfold = GroupKFold(n_splits=k, random_state=random_state)
#    group = dataset.Patient_ID.values
#    x = dataset.Path.values
#    y = dataset.Label.values
    
def match_patient_split(dataset, split):
    """Recover previously saved patient split
    """
    train_patients, valid_patients, test_patients = split
    indices = np.arange(len(dataset))
    train_idx = indices[np.any(dataset.patients[:, np.newaxis] ==
                               train_patients[np.newaxis], axis=1)]
    valid_idx = indices[np.any(dataset.patients[:, np.newaxis] ==
                               valid_patients[np.newaxis], axis=1)]
    test_idx = indices[np.any(dataset.patients[:, np.newaxis] ==
                              test_patients[np.newaxis], axis=1)]

    return train_idx, valid_idx, test_idx


def patient_kfold(dataset, n_splits=5, random_state=0, valid_size=0.1):
    """Perform cross-validation with patient split.
    """
    indices = np.arange(len(dataset))

    patients_unique = np.unique(dataset.Patient_ID)

    skf = KFold(n_splits, shuffle=True, random_state=random_state)
    ind = skf.split(patients_unique)

    train_idx = []
    valid_idx = []
    test_idx = []

    for k, (ind_train, ind_test) in enumerate(ind):

        patients_train = patients_unique[ind_train]
        patients_test = patients_unique[ind_test]

        test_idx.append(indices[np.any(np.array(dataset.Patient_ID)[:, np.newaxis] ==
                                       np.array(patients_test)[np.newaxis], axis=1)])

        if valid_size > 0:
            patients_train, patients_valid = train_test_split(
                patients_train, test_size=valid_size, random_state=0)
            valid_idx.append(indices[np.any(np.array(dataset.Patient_ID)[:, np.newaxis] ==
                                            np.array(patients_valid)[np.newaxis], axis=1)])

        train_idx.append(indices[np.any(np.array(dataset.Patient_ID)[:, np.newaxis] ==
                                        np.array(patients_train)[np.newaxis], axis=1)])

    return train_idx, valid_idx, test_idx

def patient_kfold_variable(dataset, n_splits=5, random_state=0, valid_size=0.1, variable='country'):
    """Perform cross-validation per variable.
    """
    indices = np.arange(len(dataset))

    train_idx = []
    valid_idx = []
    test_idx = []
    classes = np.unique(dataset[variable].values)
    import pdb; pdb.set_trace()
    for k, class_ in enumerate(classes):
        test_idx.append(indices[dataset[variable].values ==
                                       class_])

        train_idx.append(indices[dataset[variable].values !=
                                       class_])
        
        train, val = train_test_split(
                train_idx[-1], test_size=valid_size, random_state=0)
        
        train_idx[-1] = train

        valid_idx.append(val)

    return train_idx, valid_idx, test_idx

def match_patient_kfold(dataset, splits):
    """Recover previously saved patient splits for cross-validation.
    """

    indices = np.arange(len(dataset))
    train_idx = []
    valid_idx = []
    test_idx = []

    for train_patients, valid_patients, test_patients in splits:

        train_idx.append(indices[np.any(dataset.patients[:, np.newaxis] ==
                                        train_patients[np.newaxis], axis=1)])
        valid_idx.append(indices[np.any(dataset.patients[:, np.newaxis] ==
                                        valid_patients[np.newaxis], axis=1)])
        test_idx.append(indices[np.any(dataset.patients[:, np.newaxis] ==
                                       test_patients[np.newaxis], axis=1)])

    return train_idx, valid_idx, test_idx

def exists(x):
    return x != None
