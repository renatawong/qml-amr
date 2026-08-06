import subprocess
import os

for filename in os.listdir("dataset/by_antibiotic_12comorbdims"):

    drug = filename.replace(".csv", "")
    raw = subprocess.run(f"python ./quantum_ml_pipeline.py -d {drug}", shell=True, capture_output=True, text=True).stdout.strip()
    over = subprocess.run(f"python ./quantum_ml_pipeline.py -d {drug} -o 1", shell=True, capture_output=True, text=True).stdout.strip()
    under = subprocess.run(f"python ./quantum_ml_pipeline.py -d {drug} -u 1", shell=True, capture_output=True, text=True).stdout.strip()
    
    result_raw = subprocess.run(f"python ./qml_evaluation.py -d {drug} -m {raw}", shell=True, capture_output=True, text=True).stdout.strip()
    result_over = subprocess.run(f"python ./qml_evaluation.py -d {drug} -m {over}", shell=True, capture_output=True, text=True).stdout.strip()
    result_under = subprocess.run(f"python ./qml_evaluation.py -d {drug} -m {under}", shell=True, capture_output=True, text=True).stdout.strip()
    
    with open(f"QML Results/{drug}.txt", "w+") as file:
        file.write(f"{result_raw}\n{result_over}\n{result_under}")
