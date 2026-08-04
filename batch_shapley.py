import os
import subprocess

for filename in os.listdir("CML Models"):
    drug = filename.split("_")[0]
    model_code = filename.replace(".model","")
    if "12" in filename:
        subprocess.run(f"python ./shapley.py -m {model_code} -d {drug} -r 1", shell=True)
    else:
        subprocess.run(f"python ./shapley.py -m {model_code} -d {drug}", shell=True)
for filename in os.listdir("QML Models"):
    drug = filename.split("_")[0]
    model_code = filename.replace(".model","")
    subprocess.run(f"python ./shapley.py -m {model_code} -d {drug} -q 1 -r 1", shell=True)
