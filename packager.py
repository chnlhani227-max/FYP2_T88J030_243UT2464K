import os
import zipfile

def create_zip(algo_name):
    base = "deployment_builds"
    zip_path = f"{base}/{algo_name}_Deploy.zip"
    try:
        with zipfile.ZipFile(zip_path, 'w') as z:
            z.write(f"{base}/README.txt", "README.txt")
            if os.path.exists(f"{base}/{algo_name}.tflite"):
                z.write(f"{base}/{algo_name}.tflite", f"{algo_name}.tflite")
            elif os.path.exists(f"{base}/{algo_name}.joblib"):
                z.write(f"{base}/{algo_name}.joblib", f"{algo_name}.joblib")
    except:
        pass
    return zip_path

with open("deployment_builds/README.txt", "w") as f:
    f.write("Copyright © Dr Subar - FIST MMU Melaka 2025\n")
    f.write("Deployment package for FYP model.\n")
