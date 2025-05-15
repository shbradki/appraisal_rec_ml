import json
import pandas as pd
import os

def safe_abs(val):
    try:
        return abs(val)
    except:
        return None

def build_training_data_from_cleaned(cleaned_file):
    with open(cleaned_file, "r") as f:
        data = json.load(f)

    rows = []

    for appraisal in data["appraisals"]:
        subject = appraisal["subject"]
        order_id = appraisal.get("orderID", "UNKNOWN")

        # Track comp addresses (lowercased)
        comp_addresses = {
            comp.get("address", "").strip().lower()
            for comp in appraisal.get("comps", [])
        }

        used_addresses = set()

        # Include the comps explicitly
        for comp in appraisal.get("comps", []):
            try:
                address = comp.get("address", "").strip()
                if not address or address.lower() in used_addresses:
                    continue

                row = {
                    "orderID": order_id,
                    "candidate_address": address,
                    "is_comp": 1,

                    "subject_address": subject['address'],

                    "bath_score_diff": comp.get('bath_score_diff'),
                    "full_baths_diff": comp.get('full_baths_diff'),
                    "half_baths_diff": comp.get('half_baths_diff'),
                    "room_count_diff": comp.get('room_count_diff'),
                    "bedrooms_diff": comp.get('bedrooms_diff'),
                    "effective_age_diff": comp.get('effective_age_diff'),
                    "subject_age_diff": comp.get('subject_age_diff'),
                    "lot_size_sf_diff": comp.get('lot_size_diff_sf'),
                    "gla_diff": comp.get('gla_diff'),

                    "abs_bath_score_diff": safe_abs(comp.get("bath_score_diff")),
                    "abs_full_bath_diff": safe_abs(comp.get("full_baths_diff")),
                    "abs_half_bath_diff": safe_abs(comp.get("half_baths_diff")),
                    "abs_room_count_diff": safe_abs(comp.get("room_count_diff")),
                    "abs_bedrooms_diff": safe_abs(comp.get("bedrooms_diff")),
                    "abs_effective_age_diff": safe_abs(comp.get("effective_age_diff")),
                    "abs_subject_age_diff": safe_abs(comp.get("subject_age_diff")),
                    "abs_lot_size_sf_diff": safe_abs(comp.get('lot_size_diff_sf')),
                    "abs_gla_diff": safe_abs(comp.get('gla_diff')),

                    "same_property_type": comp.get("same_property_type", None),
                    "sold_recently": comp.get("sold_recently", None),
                }

                rows.append(row)
                used_addresses.add(address.lower())

            except Exception as e:
                print(f"⚠️ Skipping bad comp in order {order_id}: {e}")
                continue

        # Include properties from candidate pool
        for prop in appraisal.get("properties", []):
            try:
                address = prop.get("address", "").strip()
                if not address or address.lower() in used_addresses:
                    continue

                label = int(address.lower() in comp_addresses)

                row = {
                    "orderID": order_id,
                    "candidate_address": address,
                    "is_comp": label,

                    "subject_address": subject['address'],

                    "bath_score_diff": prop.get('bath_score_diff'),
                    "full_baths_diff": prop.get('full_baths_diff'),
                    "half_baths_diff": prop.get('half_baths_diff'),
                    "room_count_diff": prop.get('room_count_diff'),
                    "bedrooms_diff": prop.get('bedrooms_diff'),
                    "effective_age_diff": prop.get('effective_age_diff'),
                    "subject_age_diff": prop.get('subject_age_diff'),
                    "lot_size_sf_diff": prop.get('lot_size_diff_sf'),
                    "gla_diff": prop.get('gla_diff'),

                    "abs_bath_score_diff": safe_abs(prop.get("bath_score_diff")),
                    "abs_full_bath_diff": safe_abs(prop.get("full_baths_diff")),
                    "abs_half_bath_diff": safe_abs(prop.get("half_baths_diff")),
                    "abs_room_count_diff": safe_abs(prop.get("room_count_diff")),
                    "abs_bedrooms_diff": safe_abs(prop.get("bedrooms_diff")),
                    "abs_effective_age_diff": safe_abs(prop.get("effective_age_diff")),
                    "abs_subject_age_diff": safe_abs(prop.get("subject_age_diff")),
                    "abs_lot_size_sf_diff": safe_abs(prop.get('lot_size_diff_sf')),
                    "abs_gla_diff": safe_abs(prop.get('gla_diff')),

                    "same_property_type": prop.get("same_property_type", None),
                    "sold_recently": prop.get("sold_recently", None),
                }

                rows.append(row)
                used_addresses.add(address.lower())

            except Exception as e:
                print(f"⚠️ Skipping bad property in order {order_id}: {e}")
                continue

    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    input_file = "feature_engineered_appraisals_dataset.json"
    output_file = "training_data.csv"

    if not os.path.exists(input_file):
        raise FileNotFoundError(f" Input file not found: {input_file}")

    df = build_training_data_from_cleaned(input_file)
    df.to_csv(output_file, index=False)
    print(f"Saved training data to '{output_file}' with shape: {df.shape}")
