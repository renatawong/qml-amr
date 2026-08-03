import subprocess
import os

for filename in os.listdir("dataset/by_antibiotic"):

    drug = filename.replace(".csv", "")
    result_raw = subprocess.run(f"python ./classical_ml_pipeline.py -d {drug}", shell=True, capture_output=True, text=True).stdout.strip()
    result_over = subprocess.run(f"python ./classical_ml_pipeline.py -d {drug} -o 1", shell=True, capture_output=True, text=True).stdout.strip()
    result_under = subprocess.run(f"python ./classical_ml_pipeline.py -d {drug} -u 1", shell=True, capture_output=True, text=True).stdout.strip()
    result_raw11 = subprocess.run(f"python ./classical_ml_pipeline.py -d {drug} --reduce 1", shell=True, capture_output=True, text=True).stdout.strip()
    result_over11 = subprocess.run(f"python ./classical_ml_pipeline.py -d {drug} -o 1 --reduce 1", shell=True, capture_output=True, text=True).stdout.strip()
    result_under11 = subprocess.run(f"python ./classical_ml_pipeline.py -d {drug} -u 1 --reduce 1", shell=True, capture_output=True, text=True).stdout.strip()
    with open(f"CML Results/{drug}.txt", "w+") as file:
        file.write(f"{result_raw}\n{result_over}\n{result_under}\n{result_raw11}\n{result_over11}\n{result_under11}")
