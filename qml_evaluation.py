#!/usr/bin/env python
# coding: utf-8
import argparse

from sklearn.compose import ColumnTransformer

parser = argparse.ArgumentParser(description="Quantum ML evaluation pipeline")
parser.add_argument("-m", "--model", type=str, help="Model code to load")
parser.add_argument("-d", "--drug", type=str, help="Drug to load")
parser.add_argument("--dup", type=bool, default=False, help="Use duplicate patients?")
parser.add_argument("-a", "--amplitude", type=bool, default=False, help="Used an amplitude feature map?")
parser.add_argument("-v", "--verbose", type=bool, default=False, help="Show verbose status messages?")

args = parser.parse_args()
# In[23]:


MODEL_CODE = args.model
VERBOSE = args.verbose

# In[24]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import pandas as pd


DATASET_CODE = f"dataset/by_antibiotic"
if not args.amplitude:
    DATASET_CODE += "_12comorbdims"
if args.dup:
    DATASET_CODE += "_dup"
enc_cat_cols = ["age", "gender", "procedure_description"]
dataset = pd.read_csv(f"{DATASET_CODE}/{args.drug}.csv")

# pca = make_pipeline(
#     PCA(n_components=10)
# )
cat_cols = list(dataset.drop(columns=['Unnamed: 0', 'anon_id','order_time_jittered_utc_shifted', 'resistant','adi_score','nursing_home_visit_culture', 'last_dose_to_culture','procedure_days_culture']).columns)
scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))

encoder = ColumnTransformer(
    transformers=[('ordinal', OrdinalEncoder(), enc_cat_cols)],
    remainder='passthrough' 
)
X = dataset.drop(columns=["anon_id","order_time_jittered_utc_shifted","resistant","Unnamed: 0"]).fillna(-9999999)
X[enc_cat_cols] = X[enc_cat_cols].astype(str)
X = encoder.fit_transform(X)
# X = pca.fit_transform(X)

y = dataset["resistant"].to_numpy()


X = scaler.fit_transform(X)
# X, y = resample(X, y, n_samples=50000, replace=False, stratify=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


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

