import torch
import pandas as pd
import numpy as np
import argparse
from self_paced_ensemble import SelfPacedEnsembleClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from Contrastive_AD.train import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def train_and_evaluate_SPE(X_train, X_test, y_train, y_test, random_state):
    clf = SelfPacedEnsembleClassifier(
            n_estimators=289,
            k_bins=10,
            random_state=random_state
        ).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    aucroc = roc_auc_score(y_test, y_pred)
    score = average_precision_score(y_test, y_pred)
    print ("SelfPacedEnsemble {} | AUCROC: {:.3f} | AUPRC: {:.3f} | #Training Samples: {:d}".format(
        len(clf.estimators_), aucroc, score, sum(clf.estimators_n_training_samples_)
        ))

def read_data(args):
    print("Reading data...")
    if args.use_normalised:
        data = pd.read_csv('Data/creditcardfraud_normalised.csv')
        y = data['class'] 
        X = data.drop(['class'], axis=1)
    else:
        data = pd.read_csv('Data/creditcard.csv')
        y = data['Class'] 
        X = data.drop(['Class'], axis=1)
    print("size of datawset: {}".format(data.shape[0]))
    X = X.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.random_seed)
    print("size of trainset: {}, size of testset: {}".format(y_train.shape[0], y_test.shape[0]))
    return X_train, X_test, y_train, y_test

def main(args):
    X_train, X_test, y_train, y_test = read_data(args)
    print("===SPE===")
    train_and_evaluate_SPE(X_train, X_test, y_train, y_test, args.random_seed)

    print("==contrastive===")
    # TODO: only use normal sample in training
    contrastive_trainer_object=ContrastiveTrainer(batch_size=1024, device=device, faster_version=False)
    f_score, aucroc, auprc = contrastive_trainer_object.train_and_evaluate(X_train, X_test, y_test)
    print ("F1: {:.3f} | AUCROC: {:.3f} | AUPRC: {:.3f} ".format(f_score, aucroc, auprc))

parser = argparse.ArgumentParser()
parser.add_argument("--use_normalised", action='store_true', help="use normalised data or not")
parser.add_argument("--random_seed", type=int, default=42, help="the random seed number")
args = parser.parse_args()
main(args)
