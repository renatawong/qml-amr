#!/usr/bin/env python
# coding: utf-8
import argparse
from curses import raw
from os import device_encoding
parser = argparse.ArgumentParser(description="Quantum ML pipeline for learning antimicrobial resistance on patient EHR")
parser.add_argument("-d", "--drug", type=str, help="Drug code to learn on")
parser.add_argument("-s", "--autosave", type=bool, default=True, help="Save to file with the model code?")
parser.add_argument("-r", "--resample", type=bool, default=False, help="Resample to a smaller dataset?")
parser.add_argument("-u", "--undersample", type=bool, default=False, help="Undersample using Imbalanced Learn?")
parser.add_argument("-o", "--oversample", type=bool, default=False, help="Oversample using Imbalanced Learn?")
parser.add_argument("-c", "--count", type=int, default=5000, help="Number of samples in resample")
parser.add_argument("-f", "--featuremap", type=str, default="efficientsu2", help="Which featuremap to use?")
parser.add_argument("-t", "--continuetraining", type=bool, default=False, help="Continue model training?")
parser.add_argument("-g", "--gpuaccel", type=bool, default=False, help="Use a GPU accelerated sampler?")
parser.add_argument("--debug", type=bool, default=False, help="Show all debug print statements?")
parser.add_argument("-v", "--verbose", type=bool, default=False, help="Show verbose status messages?")

args = parser.parse_args()

VERBOSE = args.verbose
DEBUG = args.debug
# In[15]:


import pandas as pd
FEATURE_MAP = args.featuremap
if FEATURE_MAP == "amplitude":
    dataset = pd.read_csv(f"dataset/by_antibiotic/{args.drug}.csv")
    features = 512
else:
    dataset = pd.read_csv(f"dataset/by_antibiotic_12comorbdims/{args.drug}.csv")
    features = 20
if VERBOSE:
    print("loaded dataset! (1/9)")


# In[16]:

if DEBUG:
    print(dataset.columns)


# In[ ]:


AUTO_SAVE = args.autosave
MODEL_CODE = f"{args.drug}_{features}features_{FEATURE_MAP}map_realamplitudes3reps"
RESAMPLE = args.resample
SAMPLES = args.count
OVERSAMPLE = args.oversample
UNDERSAMPLE = args.undersample
CONTINUE_TRAINING = args.continuetraining
if OVERSAMPLE and UNDERSAMPLE:
    raise ValueError("You cannot both oversample and undersample!")

if RESAMPLE:
    MODEL_CODE += f"_{SAMPLES}samples"
if OVERSAMPLE:
    MODEL_CODE += "_oversampled"
if UNDERSAMPLE:
    MODEL_CODE += "_undersampled"

GPU_ACCEL = args.gpuaccel

if GPU_ACCEL:
    MODEL_CODE += "_gpu"

# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC
from sklearn.utils import resample
if UNDERSAMPLE:
    rs = RandomUnderSampler(sampling_strategy=1)

if OVERSAMPLE:
    cat_cols = list(dataset.drop(columns=['Unnamed: 0', 'anon_id','order_time_jittered_utc_shifted', 'resistant','adi_score','nursing_home_visit_culture', 'last_dose_to_culture','procedure_days_culture']).columns)
    rs = SMOTENC(categorical_features=cat_cols, sampling_strategy=1)


# pca = make_pipeline(
#     PCA(n_components=9)
# )

encoder = OrdinalEncoder()
X = dataset.drop(columns=["anon_id","order_time_jittered_utc_shifted","resistant", 'Unnamed: 0'])
if features == 512:
    X = X.drop(columns=['Unnamed: 0.3', 'Unnamed: 0.2', 'Unnamed: 0.1'])

X = X.fillna(-9999999).astype(str)
# X = pca.fit_transform(X)

y = dataset["resistant"]

if OVERSAMPLE or UNDERSAMPLE:
    X_rs, y_rs = rs.fit_resample(X, y)
    y_rs = y_rs.to_numpy()
    X_rs = encoder.fit_transform(X_rs)

X = encoder.fit_transform(X)

# X = np.nan_to_num(X, -9999999)

y=y.to_numpy()




if RESAMPLE:
    X, y = resample(X, y, n_samples=SAMPLES, replace=False, stratify=y)
    if OVERSAMPLE or UNDERSAMPLE:
        X_rs, y_rs = resample(X_rs, y_rs, n_samples=SAMPLES, replace=False, stratify=y_rs)

if OVERSAMPLE or UNDERSAMPLE:
    X_train, X_test, y_train, y_test = train_test_split(X_rs, y_rs, test_size=0.2, stratify=y_rs)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_train_us, X_test_us, y_train_us, y_test_us = train_test_split(X, y, test_size=0.2, stratify=y)
if VERBOSE:
    print("Completed data preparation (including over/undersampling if applicable) (2/9)")

# In[ ]:





# In[19]:


from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import efficient_su2
from qiskit_machine_learning.circuit.library import raw_feature_vector

num_features = X_train.shape[1]

if FEATURE_MAP == "efficientsu2":
    reps = 1
    num_qubits = int(num_features / (2*(reps+1))) # enable for efficient su2
    feature_map = efficient_su2(num_qubits=num_qubits, reps=reps, entanglement="full", parameter_prefix="f")
elif FEATURE_MAP == "zz":
    num_qubits = num_features # enable for zzfeaturemap

    feature_map = ZZFeatureMap(num_features)  
elif FEATURE_MAP == "amplitude":
    pca = PCA(n_components=512)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    X_train_us = pca.transform(X_train_us)
    X_test_us = pca.transform(X_test_us)

    feature_map = raw_feature_vector(512)
else:
    raise ValueError("Invalid feature map! Please check spelling.")
if VERBOSE:
    print("Created feature map! (3/9)")

# In[20]:


from qiskit.circuit.library import real_amplitudes
from qiskit.circuit.library import efficient_su2

# log2features = int(np.log2(num_features)) # use with raw feature map

if FEATURE_MAP == "amplitude":
    ansatz = real_amplitudes(num_qubits=9, reps=3, parameter_prefix="a")
else:
    ansatz = real_amplitudes(num_qubits=num_qubits, reps=3, parameter_prefix="a")
# ansatz = efficient_su2(num_qubits=int(np.log2(num_features)), reps=3)

OPTIMIZER_ITERS = 100 #40 for efficient_su2 ansatz, 100 for real_amplitudes

if VERBOSE:
    print("Created ansatz! (4/9)")
# In[21]:


from qiskit_machine_learning.optimizers import COBYLA

optimizer = COBYLA(maxiter=OPTIMIZER_ITERS)
if VERBOSE:
    print("Created optimizer! (5/9)")

# In[22]:


from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_aer.primitives import SamplerV2 as GPUSampler
from qiskit_aer import AerSimulator
if GPU_ACCEL:
    backend = AerSimulator(device="GPU", method="statevector", cuQuantum_enable=True)
    sampler = GPUSampler(backend=backend)
else:
    sampler = Sampler()
if VERBOSE:
    print("Created sampler! (6/9)")

# In[23]:


from matplotlib import pyplot as plt
from IPython.display import clear_output

objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)


def callback_graph(weights, obj_func_eval):

    # clear_output(wait=True)
    global VERBOSE
    if VERBOSE:
        objective_func_vals.append(obj_func_eval)
        # plt.title("Objective function value against iteration")
        # plt.xlabel("Iteration")
        # plt.ylabel("Objective function value")
        # plt.plot(range(len(objective_func_vals)), objective_func_vals)
        # plt.show()
        print(len(objective_func_vals))

# In[24]:
import logging

# Hard-override logging outputs across all Qiskit modules globally
logging.basicConfig(level=logging.ERROR)
logging.getLogger("qiskit").setLevel(logging.ERROR)
logging.getLogger("qiskit_machine_learning").setLevel(logging.ERROR)

import time
from qiskit_machine_learning.algorithms.classifiers import VQC

if CONTINUE_TRAINING:
    vqc = VQC.from_dill(f"QML Models/{MODEL_CODE}.model")
    vqc.warm_start = True
    vqc.optimizer = optimizer
    vqc.sampler = sampler
else:
    vqc = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=callback_graph
    )


# clear objective value history
objective_func_vals = []

if VERBOSE:
    print("Starting training! (7/9)")

start = time.time()
vqc.fit(X_train, y_train)
elapsed = time.time() - start

if VERBOSE:
    print(f"Finished training! Training time: {round(elapsed)} seconds (8/9)")


# In[25]:


if AUTO_SAVE:
    vqc.to_dill(f"QML Models/{MODEL_CODE}.model")
    if VERBOSE:
        print("Saved model! (9/9)")

print(MODEL_CODE)
# In[26]:


# train_score_q4 = vqc.score(X_train, y_train)
# test_score_q4 = vqc.score(X_test, y_test)

# print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
# print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")


# In[27]:


# from sklearn.metrics import confusion_matrix

# y_pred = vqc.predict(X_test)

# tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel().tolist()
# print(f"True negatives: {tn} ({tn / len(y_test)})\n False positives: {fp} ({fp / len(y_test)})\n False negatives: {fn} ({fn / len(y_test)})\n True positives: {tp} ({tp / len(y_test)})")

