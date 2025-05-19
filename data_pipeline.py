import os
import subprocess
import json

def normalize_address(address):
    if not address:
        return None
    return address.lower().strip()\
        .replace("drive", "dr")\
        .replace("road", "rd")\
        .replace("street", "st")\
        .replace("avenue", "ave")

def should_run_geocoding():
    cache_path = "geocoded_addresses.json"
    data_path = "cleaned_appraisals_dataset.json"

    if not os.path.exists(cache_path):
        return True

    with open(cache_path, "r") as f:
        cached = set(json.load(f).keys())

    with open(data_path, "r") as f:
        data = json.load(f)

    needed = set()
    for appraisal in data.get("appraisals", []):
        all_addresses = (
            [appraisal.get("subject", {}).get("address", "")]
            + [comp.get("address", "") for comp in appraisal.get("comps", [])]
            + [prop.get("address", "") for prop in appraisal.get("properties", [])]
        )

        for raw_addr in all_addresses:
            norm = normalize_address(raw_addr)
            if norm:
                needed.add(norm)

    missing = [addr for addr in needed if addr not in cached]

    return len(missing) > 0

def run(script):
    print(f"\nRunning {script} ...")
    subprocess.run(["/usr/local/bin/python3.12", script], check=True)

# Stage 1: Clean raw data
run("clean_initial_data.py")

# Stage 2: Run geocoder only if necessary
if should_run_geocoding():
    run("geocode_all_addresses.py")
else:
    print("All addresses already geocoded — skipping.")

# Stage 3–6: Always re-run
run("features.py")
run("training_data.py")
run("train_model.py")
run("top3_explanations.py")
