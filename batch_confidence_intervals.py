import os
import subprocess

for filename in os.listdir("CML Models"):
    if "_" not in filename:
        continue
    result = subprocess.run(f"python3.11 ./confidence_intervals.py -m {filename.replace(".csv","")}")
    with open(f"CML Results/confidence/{filename.split("_")[0]}.txt", "a+") as file:
        if "over" in filename:
            file.write("Oversampled")
        elif "under" in filename:
            file.write("Undersampled")
        else:
            file.write("Raw")
        file.write(result)
for filename in os.listdir("QML Models"):
    if "_" not in filename:
        continue
    result = subprocess.run(f"python3.11 ./confidence_intervals.py -m {filename.replace(".csv","")} -q 1")
    with open(f"QML Results/confidence/{filename.split("_")[0]}.txt", "a+") as file:
        if "over" in filename:
            file.write("Oversampled")
        elif "under" in filename:
            file.write("Undersampled")
        else:
            file.write("Raw")
        file.write(result)