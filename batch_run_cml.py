import subprocess
import os
import argparse
parser = argparse.ArgumentParser(description="Quantum ML pipeline for learning antimicrobial resistance on patient EHR")
parser.add_argument("-s", "--shortlist", type=bool, default=False, help="Use a drug shortlist? Leave unset to use all.")

shortlist = ['AMP.csv', 'CFR.csv', 'CXM.csv', 'FOF.csv', 'GEN.csv', 'LZD.csv', 'MFG.csv', 'MIN.csv', 'PEN.csv']

args = parser.parse_args()

if args.shortlist:
    drugs_list = shortlist
else:
    drugs_list = os.listdir("dataset/by_antibiotic")

for filename in drugs_list:

    drug = filename.replace(".csv", "")
    result_raw = subprocess.run(f"python ./classical_ml_pipeline.py -d {drug} --dup 1", shell=True, capture_output=True, text=True).stdout.strip()
    result_over = subprocess.run(f"python ./classical_ml_pipeline.py -d {drug} -o 1 --dup 1", shell=True, capture_output=True, text=True).stdout.strip()
    result_under = subprocess.run(f"python ./classical_ml_pipeline.py -d {drug} -u 1 --dup 1", shell=True, capture_output=True, text=True).stdout.strip()
    result_raw11 = subprocess.run(f"python ./classical_ml_pipeline.py -d {drug} --reduce 1 --dup 1", shell=True, capture_output=True, text=True).stdout.strip()
    result_over11 = subprocess.run(f"python ./classical_ml_pipeline.py -d {drug} -o 1 --reduce 1 --dup 1", shell=True, capture_output=True, text=True).stdout.strip()
    result_under11 = subprocess.run(f"python ./classical_ml_pipeline.py -d {drug} -u 1 --reduce 1 --dup 1", shell=True, capture_output=True, text=True).stdout.strip()
    with open(f"CML Results/{drug}.txt", "w+") as file:
        file.write(f"{result_raw}\n{result_over}\n{result_under}\n{result_raw11}\n{result_over11}\n{result_under11}")
