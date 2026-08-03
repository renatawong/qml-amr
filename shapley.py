#!/usr/bin/env python
# coding: utf-8
import argparse
import pickle
parser = argparse.ArgumentParser(description="Generic Shapley pipeline")
parser.add_argument("-m", "--model", type=str, help="Model code to load")
parser.add_argument("-d", "--drug", type=str, help="Drug to load")
parser.add_argument("-q", "--quantum", type=bool, default=False, help="Load quantum model? Set to false for classical models.")
parser.add_argument("-r", "--reduced", type=bool, default=False, help="Load 12 comorbidity dimensions dataset?")
parser.add_argument("-v", "--verbose", type=bool, default=False, help="Show verbose status messages?")

args = parser.parse_args()
# In[ ]:

QUANTUM = args.quantum
DRUG = args.drug
MODEL_CODE = args.model
REDUCED = args.reduced
VERBOSE = args.verbose



# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

if REDUCED:
    dataset = pd.read_csv(f"dataset/by_antibiotic_12comorbdims/{DRUG}.csv")
else:
    dataset = pd.read_csv(f"dataset/by_antibiotic/{DRUG}.csv")
# pca = make_pipeline(
#     PCA(n_components=10)
# )

encoder = OrdinalEncoder()
X = dataset.drop(columns=["anon_id","order_time_jittered_utc_shifted","resistant","Unnamed: 0"])
if not REDUCED:
    X = X.drop(columns=['Unnamed: 0.3', 'Unnamed: 0.2', 'Unnamed: 0.1'])
X = encoder.fit_transform(X)
X = np.nan_to_num(X, nan=-99999)
# X = pca.fit_transform(X)

y = dataset["resistant"].to_numpy()

#print(X.shape)
X = scaler.fit_transform(X, y)
#print(X.shape)
# X, y = resample(X, y, n_samples=50000, replace=False, stratify=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
#print(X_test.shape, X_train.shape)
if VERBOSE:
    print("Loaded dataset! (1/6)")

# In[ ]:

if QUANTUM:
    from qiskit_machine_learning.algorithms.classifiers import VQC

    clf = VQC.from_dill(f"QML Models/{MODEL_CODE}.model")
else:
    import joblib

    model = joblib.load(f"CML Models/{MODEL_CODE}.model")
    clf = model


if VERBOSE:
    print("Loaded model! (2/6)")
# In[ ]:


import shap

background_profile = np.mean(X_train, axis=0).reshape(1, -1)
exp = shap.PermutationExplainer(clf.predict_proba, background_profile)

if VERBOSE:
    print("Created explainer! (3/6)")
# In[ ]:

if VERBOSE:
    print("Starting shap calculation! (4/6)")
#print(X_test.shape)
if REDUCED:
    shapley = exp(X_test, silent=(not VERBOSE))
else:
    shapley = exp(X_test, max_evals=1037, silent=(not VERBOSE))
if VERBOSE:
    print("Completed shap calculation! (5/6)")
if QUANTUM:
    with open(f"QShapley/{DRUG}.sav", "wb") as f:
        pickle.dump(shapley, f)
else:
    code = f"CShapley/{DRUG}"
    if REDUCED:
        code += "_reduced"
    with open(f"{code}.sav", "wb") as f:
        pickle.dump(shapley, f)
if VERBOSE:
    print("Saved to file! (6/6)")
    shap.plots.beeswarm(shapley[:,:,1], max_display=15)


# In[ ]:




