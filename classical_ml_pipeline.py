#!/usr/bin/env python
# coding: utf-8
import argparse
parser = argparse.ArgumentParser(description="Classical ML pipeline for learning antimicrobial resistance on patient EHR")
parser.add_argument("-d", "--drug", type=str, help="Drug code to learn on")
parser.add_argument("-s", "--autosave", type=bool, default=True, help="Save to file with the model code?")
parser.add_argument("-r", "--resample", type=bool, default=False, help="Resample to a smaller dataset?")
parser.add_argument("-u", "--undersample", type=bool, default=False, help="Undersample using Imbalanced Learn?")
parser.add_argument("-o", "--oversample", type=bool, default=False, help="Oversample using Imbalanced Learn?")
parser.add_argument("-c", "--count", type=int, default=5000, help="Number of samples in resample")
parser.add_argument("--reduce", type=bool, default=False, help="Use the reduced comorbidity dimensionality dataset?")
parser.add_argument("--dup", type=bool, default=False, help="Use duplicate patients?")
parser.add_argument("--debug", type=bool, default=False, help="Show all debug print statements?")
parser.add_argument("-v", "--verbose", type=bool, default=False, help="Show verbose status messages?")

args = parser.parse_args()

VERBOSE = args.verbose
DEBUG = args.debug

# In[33]:


import pandas as pd
dataset_code = f"dataset/by_antibiotic"
if args.reduce:
    dataset_code += "_12comorbdims"
if args.dup:
    dataset_code += "_dup"
dataset = pd.read_csv(f"{dataset_code}/{args.drug}.csv")

if VERBOSE:
    print("loaded dataset! (1/6)")
# In[34]:

if DEBUG:
    print(dataset.columns)


# In[35]:


AUTO_SAVE = args.autosave
MODEL_CODE = f"{args.drug}_hgbc_alpha0.01_maxiter5000"
RESAMPLE = args.resample
SAMPLES = args.count
OVERSAMPLE = args.oversample
UNDERSAMPLE = args.undersample

if OVERSAMPLE and UNDERSAMPLE:
    raise ValueError("You cannot both oversample and undersample!")

if args.reduce:
    MODEL_CODE += "_12features"
else:
    MODEL_CODE += "_521features"
if RESAMPLE:
    MODEL_CODE += f"_{SAMPLES}samples"
if OVERSAMPLE:
    MODEL_CODE += "_oversampled"
if UNDERSAMPLE:
    MODEL_CODE += "_undersampled"
if args.dup:
    MODEL_CODE += "_dup"

# In[36]:

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC

clf = make_pipeline(MinMaxScaler(feature_range=(-np.pi, np.pi)), HistGradientBoostingClassifier(learning_rate=0.01, max_iter=5000,class_weight="balanced"))
# clf = make_pipeline(StandardScaler(), SVC(gamma="auto",class_weight="balanced"))

if VERBOSE:
    print("created scaler-model pipeline! (2/6)")
# In[37]:



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC
from sklearn.utils import resample
from sklearn.compose import ColumnTransformer
if UNDERSAMPLE:
    rs = RandomUnderSampler(sampling_strategy=1)
if args.reduce:
    cat_cols = ['age', 'gender', 'procedure_description','wards_int']
else:
    cat_cols = list(dataset.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'anon_id','order_time_jittered_utc_shifted', 'resistant','adi_score','nursing_home_visit_culture', 'last_dose_to_culture','procedure_days_culture']).columns)
enc_cat_cols = ["age", "gender", "procedure_description"]

if DEBUG:
    print(enc_cat_cols)
    print(dataset[enc_cat_cols].dtypes)
if OVERSAMPLE:
    if DEBUG:
        print(cat_cols)
    rs = SMOTENC(categorical_features=cat_cols, sampling_strategy=1)


# pca = make_pipeline(
#     PCA(n_components=9)
# )

encoder = ColumnTransformer(
    transformers=[('ordinal', OrdinalEncoder(), enc_cat_cols)],
    remainder='passthrough' 
)
if args.reduce:
    X = dataset.drop(columns=["anon_id","order_time_jittered_utc_shifted","resistant",'Unnamed: 0']).fillna(-9999999)
else:
    X = dataset.drop(columns=["anon_id","order_time_jittered_utc_shifted","resistant",'Unnamed: 0.1', 'Unnamed: 0']).fillna(-9999999)

# X = pca.fit_transform(X)

y = dataset["resistant"]

X[enc_cat_cols] = X[enc_cat_cols].astype(str)

if OVERSAMPLE or UNDERSAMPLE:
    X_rs, y_rs = rs.fit_resample(X.astype(str), y)
    y_rs = y_rs.to_numpy()
    X_rs = encoder.fit_transform(X_rs)

X = encoder.fit_transform(X)

# X = np.nan_to_num(X, -9999999)

y=y.to_numpy()




if RESAMPLE:
    X, y = resample(X, y, n_samples=SAMPLES, replace=False, stratify=y, random_state=42)
    if OVERSAMPLE or UNDERSAMPLE:
        X_rs, y_rs = resample(X_rs, y_rs, n_samples=SAMPLES, replace=False, stratify=y_rs, random_state=42)

if OVERSAMPLE or UNDERSAMPLE:
    X_train, X_test, y_train, y_test = train_test_split(X_rs, y_rs, test_size=0.2, stratify=y_rs, random_state=42)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train_us, X_test_us, y_train_us, y_test_us = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

if VERBOSE:
    print("Completed data preparation (including over/undersampling if applicable) (3/6)")

# In[38]:

if DEBUG:
    print(len(y_train_us))


# In[39]:

if VERBOSE:
    print("training model... (4/6)")
clf.fit(X_train, y_train)
if VERBOSE:
    print("Model training complete! (5/6)")

# In[40]:


if AUTO_SAVE:
    import pickle
    with open(f"CML Models/{MODEL_CODE}.model", "wb") as file:
        pickle.dump(clf, file)

    if VERBOSE:
        print("Saved model! (6/6)")
# In[41]:


# from sklearn.metrics import accuracy_score

# y_pred = clf.predict(X_test)
# print(accuracy_score(y_test, y_pred))


# In[42]:


# print(clf.score(X_test,y_test))
# print(clf.score(X_train, y_train))


# In[43]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

probs = clf.predict_proba(X_test_us)[:, 1]

y_pred = probs >= 0.5

tn, fp, fn, tp = confusion_matrix(y_test_us, y_pred).ravel().tolist()
print(f"   {MODEL_CODE}")
print(f"    Accuracy: {accuracy_score(y_test_us, y_pred)}")
print(f"    Confusion Matrix\n     True negatives: {tn} ({tn / len(y_test_us)})\n     False positives: {fp} ({fp / len(y_test_us)})\n     False negatives: {fn} ({fn / len(y_test_us)})\n     True positives: {tp} ({tp / len(y_test_us)})")

from sklearn.metrics import roc_auc_score, average_precision_score
positive_probabilities = probs
auroc_score = roc_auc_score(y_test_us, positive_probabilities)
prauc_score = average_precision_score(y_test_us, positive_probabilities)
print(f"    AUROC: {auroc_score:.4f}")
print(f"    PRAUC: {prauc_score:.4f}")


# In[44]:

if DEBUG:
    print(sum(y_test) / len(y_test))

