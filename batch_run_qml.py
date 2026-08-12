import subprocess
import os
import argparse
parser = argparse.ArgumentParser(description="Quantum ML pipeline for learning antimicrobial resistance on patient EHR")
parser.add_argument("-g", "--gpuaccel", type=bool, default=False, help="Use a GPU accelerated sampler?")
parser.add_argument("-s", "--shortlist", type=bool, default=False, help="Use a drug shortlist? Leave unset to use all.")

shortlist = ['AMP.csv', 'CFR.csv', 'CXM.csv', 'FOF.csv', 'GEN.csv', 'LZD.csv', 'MFG.csv', 'MIN.csv', 'PEN.csv']

args = parser.parse_args()

if args.shortlist:
    drugs_list = shortlist
else:
    drugs_list = os.listdir("dataset/by_antibiotic_12comorbdims")

for filename in drugs_list:

    drug = filename.replace(".csv", "")
    if args.gpuaccel:
        raw = subprocess.run(f"python3.11 ./quantum_ml_pipeline.py -d {drug} --dup 1 -g 1", shell=True, capture_output=True, text=True).stdout.strip()
        over = subprocess.run(f"python3.11 ./quantum_ml_pipeline.py -d {drug} -o 1 --dup 1 -g 1", shell=True, capture_output=True, text=True).stdout.strip()
        under = subprocess.run(f"python3.11 ./quantum_ml_pipeline.py -d {drug} -u 1 --dup 1 -g 1", shell=True, capture_output=True, text=True).stdout.strip()
    else:
        raw = subprocess.run(f"python3.11 ./quantum_ml_pipeline.py -d {drug} --dup 1", shell=True, capture_output=True, text=True).stdout.strip()
        over = subprocess.run(f"python3.11 ./quantum_ml_pipeline.py -d {drug} -o 1 --dup 1", shell=True, capture_output=True, text=True).stdout.strip()
        under = subprocess.run(f"python3.11 ./quantum_ml_pipeline.py -d {drug} -u 1 --dup 1", shell=True, capture_output=True, text=True).stdout.strip()
    
    result_raw = subprocess.run(f"python3.11 ./qml_evaluation.py -d {drug} -m {raw} --dup 1", shell=True, capture_output=True, text=True).stdout.strip()
    result_over = subprocess.run(f"python3.11 ./qml_evaluation.py -d {drug} -m {over} --dup 1", shell=True, capture_output=True, text=True).stdout.strip()
    result_under = subprocess.run(f"python3.11 ./qml_evaluation.py -d {drug} -m {under} --dup 1", shell=True, capture_output=True, text=True).stdout.strip()
    
    with open(f"QML Results/{drug}.txt", "w+") as file:
        file.write(f"{result_raw}\n{result_over}\n{result_under}")
