import json
import pandas as pd
import os
import re

INPUT_FILE = "feature_engineered_appraisals_dataset.json"
FEEDBACK_FILE = "feedback_log.csv"
OUTPUT_FILE = "training_data.csv"
OUTPUT_WITH_FEEDBACK = "training_data_with_feedback.csv"

def safe_abs(val):
    try:
        return abs(val)
    except:
        return None

def normalize_address(address):
    address = str(address).lower().strip()
    address = re.sub(r"\b(street|st\.?)\b", "st", address)
    address = re.sub(r"\b(road|rd\.?)\b", "rd", address)
    address = re.sub(r"\b(avenue|ave\.?)\b", "ave", address)
    address = re.sub(r"\b(drive|dr\.?)\b", "dr", address)
    address = re.sub(r"\b(unit|suite|apt)\b", "", address)
    address = address.replace("-", " ").replace(",", "").replace(".", "")
    address = re.sub(r"\s+", " ", address)
    return address.strip()

def make_row(order_id, subject, candidate, address, is_comp):
    return {
        "orderID": order_id,
        "candidate_address": address,
        "is_comp": is_comp,
        "subject_address": subject.get("address"),

        "bath_score_diff": candidate.get('bath_score_diff'),
        "full_baths_diff": candidate.get('full_baths_diff'),
        "half_baths_diff": candidate.get('half_baths_diff'),
        "room_count_diff": candidate.get('room_count_diff'),
        "bedrooms_diff": candidate.get('bedrooms_diff'),
        "effective_age_diff": candidate.get('effective_age_diff'),
        "subject_age_diff": candidate.get('subject_age_diff'),
        "lot_size_sf_diff": candidate.get('lot_size_diff_sf'),
        "gla_diff": candidate.get('gla_diff'),

        "abs_bath_score_diff": safe_abs(candidate.get("bath_score_diff")),
        "abs_full_bath_diff": safe_abs(candidate.get("full_baths_diff")),
        "abs_half_bath_diff": safe_abs(candidate.get("half_baths_diff")),
        "abs_room_count_diff": safe_abs(candidate.get("room_count_diff")),
        "abs_bedrooms_diff": safe_abs(candidate.get("bedrooms_diff")),
        "abs_effective_age_diff": safe_abs(candidate.get("effective_age_diff")),
        "abs_subject_age_diff": safe_abs(candidate.get("subject_age_diff")),
        "abs_lot_size_sf_diff": safe_abs(candidate.get("lot_size_diff_sf")),
        "abs_gla_diff": safe_abs(candidate.get("gla_diff")),

        "distance_to_subject_km": candidate.get('distance_to_subject_km'),
        "same_property_type": candidate.get("same_property_type"),
        "sold_recently": candidate.get("sold_recently")
    }

def build_training_data_from_cleaned(cleaned_file):
    with open(cleaned_file, "r") as f:
        data = json.load(f)

    rows = []

    for appraisal in data["appraisals"]:
        subject = appraisal["subject"]
        order_id = str(appraisal.get("orderID", "UNKNOWN"))
        seen_addresses = set()

        # Build a lookup for comp labels
        comp_address_lookup = {
            normalize_address(comp.get("address", ""))
            for comp in appraisal.get("comps", [])
        }

        for group, label in [("comps", 1), ("properties", 0)]:
            for prop in appraisal.get(group, []):
                raw_address = prop.get("address", "")
                norm_address = normalize_address(raw_address)
                if not norm_address or norm_address in seen_addresses:
                    continue

                is_comp = 1 if label == 1 else int(norm_address in comp_address_lookup)
                rows.append(make_row(order_id, subject, prop, raw_address, is_comp))
                seen_addresses.add(norm_address)

    return pd.DataFrame(rows)

def apply_feedback(df, feedback_file):
    if not os.path.exists(feedback_file):
        print("No feedback file found. Skipping feedback integration.")
        return df

    feedback_df = pd.read_csv(feedback_file)
    if feedback_df.empty:
        print("Feedback file is empty. Skipping feedback integration.")
        return df

    # Normalize for merge
    df["orderID"] = df["orderID"].astype(str)
    df["norm_addr"] = df["candidate_address"].apply(normalize_address)

    feedback_df["orderID"] = feedback_df["orderID"].astype(str)
    feedback_df["norm_addr"] = feedback_df["candidate_address"].apply(normalize_address)

    # Merge in feedback
    merged = df.merge(
        feedback_df[["orderID", "norm_addr", "user_feedback"]],
        on=["orderID", "norm_addr"],
        how="left"
    )

    # Override is_comp with user_feedback where available
    merged["is_comp"] = merged["user_feedback"].combine_first(merged["is_comp"])

    # Drop rows where user marked the candidate as bad and it wasn't originally a comp
    drop_mask = (merged["user_feedback"] == 0) & (merged["is_comp"] == 0)
    num_dropped = drop_mask.sum()
    merged = merged[~drop_mask]

    return merged.drop(columns=["user_feedback", "norm_addr"], errors="ignore")


if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = build_training_data_from_cleaned(INPUT_FILE)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Base training data saved to: {OUTPUT_FILE} ({df.shape})")

    df_with_feedback = apply_feedback(df.copy(), FEEDBACK_FILE)
    df_with_feedback.to_csv(OUTPUT_WITH_FEEDBACK, index=False)
    print(f"Training data with feedback saved to: {OUTPUT_WITH_FEEDBACK} ({df_with_feedback.shape})")
