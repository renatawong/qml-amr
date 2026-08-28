import argparse
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from qiskit_machine_learning.algorithms.classifiers import VQC
import pandas as pd
import joblib


parser = argparse.ArgumentParser(description="Quantum ML pipeline for learning antimicrobial resistance on patient EHR")
parser.add_argument("-m", "--model", type=str, default="", help="Model Code")
parser.add_argument("-q", "--quantum", type=bool, default=False, help="Quantum?")

args = parser.parse_args()
MODEL_CODE = args.model
DRUG = MODEL_CODE.split("_")[0]
QUANTUM = args.quantum


def bootstrap_classification_ci(y_true, y_pred, metric_func=accuracy_score, alpha=0.95, n_bootstrap=1000):
    """
    Calculates the confidence interval for a classification metric using the percentile bootstrap method.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    bootstrapped_scores = []
    n_samples = len(y_true)
    
    # Seed for reproducibility
    np.random.seed(42)
    
    # 1. Resample and calculate metrics
    for _ in range(n_bootstrap):
        # Generate indices with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Calculate metric on the bootstrap sample
        score = metric_func(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        
    # 2. Calculate lower and upper percentiles
    lower_percentile = ((1.0 - alpha) / 2.0) * 100
    upper_percentile = (alpha + ((1.0 - alpha) / 2.0)) * 100
    
    lower_bound = np.percentile(bootstrapped_scores, lower_percentile)
    upper_bound = np.percentile(bootstrapped_scores, upper_percentile)
    
    return lower_bound, upper_bound
DATASET_CODE = f"dataset/by_antibiotic"
if "12" in MODEL_CODE:
    DATASET_CODE += "_12comorbdims"
DATASET_CODE += f"_dup/{DRUG}.csv"
dataset = pd.read_csv(DATASET_CODE)
enc_cat_cols = ["age", "gender", "procedure_description"]
cat_cols = list(dataset.drop(columns=['Unnamed: 0', 'anon_id','order_time_jittered_utc_shifted', 'resistant','adi_score','nursing_home_visit_culture', 'last_dose_to_culture','procedure_days_culture']).columns)
scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
encoder = ColumnTransformer(
    transformers=[('ordinal', OrdinalEncoder(), enc_cat_cols)],
    remainder='passthrough' 
)
X = dataset.drop(columns=["anon_id","order_time_jittered_utc_shifted","resistant","Unnamed: 0"]).fillna(-9999999)
X[enc_cat_cols] = X[enc_cat_cols].astype(str)
X = encoder.fit_transform(X)
y = dataset["resistant"].to_numpy()
X = scaler.fit_transform(X)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

if QUANTUM:
    vqc = VQC.from_dill(f"QML Models/{MODEL_CODE}.model")
    test_probs = vqc.predict_proba(X_test)
    test_pos_probs = test_probs[:, 1]
    y_pred = (test_pos_probs >= 0.5)
else:
    model = joblib.load(f"CML Models/{MODEL_CODE}.model")
    test_probs = model.predict_proba(X_test)
    test_pos_probs = test_probs[:, 1]
    y_pred = (test_pos_probs >= 0.5)

lower, upper = bootstrap_classification_ci(
        y_true=y_test, 
        y_pred=y_pred, 
        metric_func=accuracy_score, 
        alpha=0.95, 
        n_bootstrap=1000
    )
print(f"{lower:.3f},{upper:.3f}")