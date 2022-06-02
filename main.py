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
    return aucroc, score

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
    aucroc = np.zeros(args.runs)
    auprc = np.zeros(args.runs)  
    for run in range(args.runs):
        print("run: ", run)
        aucroc[run], auprc[run] = train_and_evaluate_SPE(X_train, X_test, y_train, y_test, args.random_seed)
        print("AUCROC: %.4f | AUPRC: %.4f " % (aucroc[run], auprc[run]))
    mean_aucroc = np.mean(aucroc)
    mean_auprc = np.mean(auprc)
    print("average AUC-ROC: %.4f, average AUC-PR: %.4f" % (mean_aucroc, mean_auprc))   
    
    print("==contrastive===")
    aucroc = np.zeros(args.runs)
    auprc = np.zeros(args.runs)  
    for run in range(args.runs):
        print("run: ", run)
        contrastive_trainer_object=ContrastiveTrainer(batch_size=512, device=device, faster_version=False)
        f_score, aucroc[run], auprc[run] = contrastive_trainer_object.train_and_evaluate(X_train, X_test, y_test)
        print("AUCROC: %.4f | AUPRC: %.4f " % (aucroc[run], auprc[run]))
    mean_aucroc = np.mean(aucroc)
    mean_auprc = np.mean(auprc)
    print("average AUC-ROC: %.4f, average AUC-PR: %.4f" % (mean_aucroc, mean_auprc))   

parser = argparse.ArgumentParser()
parser.add_argument("--use_normalised", action='store_true', help="use normalised data or not")
parser.add_argument("--random_seed", type=int, default=42, help="the random seed number")
parser.add_argument("--runs", type=int, default=1)
args = parser.parse_args()
main(args)
