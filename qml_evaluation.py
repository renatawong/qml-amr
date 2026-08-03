#!/usr/bin/env python
# coding: utf-8
import argparse
parser = argparse.ArgumentParser(description="Quantum ML evaluation pipeline")
parser.add_argument("-m", "--model", type=str, help="Model code to load")
parser.add_argument("-d", "--drug", type=str, help="Drug to load")
parser.add_argument("-v", "--verbose", type=bool, default=False, help="Show verbose status messages?")

args = parser.parse_args()
# In[23]:


MODEL_CODE = args.model
VERBOSE = args.verbose

# In[24]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

dataset = pd.read_csv(f"dataset/by_antibiotic_12comorbdims/{args.drug}.csv")

# pca = make_pipeline(
#     PCA(n_components=10)
# )

encoder = OrdinalEncoder()
X = dataset.drop(columns=["anon_id","order_time_jittered_utc_shifted","resistant","Unnamed: 0"])
X = encoder.fit_transform(X)
X = np.nan_to_num(X, nan=-99999)
# X = pca.fit_transform(X)

y = dataset["resistant"].to_numpy()


X = scaler.fit_transform(X, y)
# X, y = resample(X, y, n_samples=50000, replace=False, stratify=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)


# In[25]:

if VERBOSE:
    print(np.mean(y))


# In[26]:


from qiskit_machine_learning.algorithms.classifiers import VQC

vqc = VQC.from_dill(f"QML Models/{MODEL_CODE}.model")


# In[27]:


test_probs = vqc.predict_proba(X_test)
train_probs = vqc.predict_proba(X_train)

test_pos_probs = test_probs[:, 1]
train_pos_probs = train_probs[:, 1]


# In[28]:


# train_score_q4 = vqc.score(X_train, y_train)
# test_score_q4 = vqc.score(X_test, y_test)

# print(f"Train accuracy: {train_score_q4:.2f}")
# print(f"Test accuracy:  {test_score_q4:.2f}")


# In[29]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

y_pred_test = (test_pos_probs >= 0.5)
y_pred_train = (train_pos_probs >= 0.5)

test_acc = accuracy_score(y_test, y_pred_test)
train_acc = accuracy_score(y_train, y_pred_train)
print(MODEL_CODE)
print(f" Test accuracy: {test_acc}\n Train accuracy: {train_acc}")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel().tolist()
print(f" Confusion Matrix\n  True negatives: {tn} ({tn / len(y_test)})\n  False positives: {fp} ({fp / len(y_test)})\n  False negatives: {fn} ({fn / len(y_test)})\n  True positives: {tp} ({tp / len(y_test)})")


# In[30]:


from sklearn.metrics import roc_auc_score, average_precision_score
auroc_score = roc_auc_score(y_test, test_pos_probs)
prauc_score = average_precision_score(y_test, test_pos_probs)
print(f" AUROC: {auroc_score:.4f}")
print(f" PRAUC: {prauc_score:.4f}")

