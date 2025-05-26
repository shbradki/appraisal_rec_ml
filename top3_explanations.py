import shap
import xgboost as xgb
import pandas as pd
import numpy as np
from openai import OpenAI
import os
from tqdm import tqdm
import json

# Load API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set.")
client = OpenAI(api_key=api_key)

# Load model and data
model = xgb.Booster()
model.load_model("xgb_rank_model.json")

with open("feature_engineered_appraisals_dataset.json") as f:
    raw_data = json.load(f)

data_file = (
    "training_data_with_feedback.csv"
    if os.path.exists("feedback_log.csv") and os.path.getsize("feedback_log.csv") > 0
    else "training_data.csv"
)

df = pd.read_csv(data_file)
print(f"Using training data: {data_file}")

# Feature columns 
feature_cols = [
    'room_count_diff', 'bedrooms_diff', 'effective_age_diff',
    'subject_age_diff', 'gla_diff', 
    'abs_bath_score_diff', 
    'abs_room_count_diff', 'abs_effective_age_diff',
    'abs_subject_age_diff', 'abs_gla_diff',
    'same_property_type', 'sold_recently', 'lot_util_diff', 'gla_per_bedroom_diff',
    'condition_diff', 'abs_basement_score_diff', 'basement_score_diff'
    
    # EXCLUDING BECAUSE OF SHAP ANALYSIS
    # 'abs_bedrooms_diff', 'abs_full_bath_diff', 'abs_half_bath_diff',
    # 'distance_to_subject_km', 'abs_lot_size_diff_acre', 'lot_size_diff_acre'
    # 'bath_score_diff', 'full_baths_diff', 'half_baths_diff',
]

df[feature_cols] = df[feature_cols].astype(float)

# Lookup actual property info 
def find_raw_values(order_id, candidate_address):
    for appraisal in raw_data["appraisals"]:
        if str(appraisal.get("orderID")) != str(order_id):
            continue
        subject = appraisal.get("subject", {})
        subject_vals = {
            "subject_bath_score": subject.get("bath_score"),
            "subject_num_full_baths": subject.get("num_full_baths"),
            "subject_num_half_baths": subject.get("num_half_baths"),
            "subject_bedrooms": subject.get('num_beds'),
            "subject_gla": subject.get("gla"),
            "subject_lot_size_sf": subject.get("lot_size_sf"),
            "subject_property_type": subject.get("property_type"),
            "subject_condition": subject.get("condition")
        }
        for group in ("comps", "properties"):
            for prop in appraisal.get(group, []):
                if prop.get("address", "").strip().lower() == candidate_address.strip().lower():
                    return subject_vals | {
                        "candidate_bath_score": prop.get("bath_score"),
                        "candidate_num_full_baths": prop.get("num_full_baths"),
                        "candidate_num_half_baths": prop.get("num_half_baths"),
                        "candidate_bedrooms": prop.get('num_beds'),
                        "candidate_gla": prop.get("gla"),
                        "candidate_lot_size_sf": prop.get("lot_size_sf"),
                        "candidate_property_type": prop.get("property_type"),
                        "candidate_close_price": prop.get("sale_price"),
                        "candidate_condition": prop.get("condition")
                    }
    return subject_vals

# GPT explanation 
def gpt_explanation(score, pos_feats, neg_feats, candidate_address, subject_address):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {
                    "role": "system",
                    "content": (
                        """You are a real estate appraisal assistant. Your job is to explain why a machine learning model ranked a 
                            candidate property as more or less comparable to a subject property.

                            The model uses feature differences (e.g., size difference, age difference) between the candidate and subject. 
                            Positive SHAP values mean the feature made the candidate more similar (better match), 
                            while negative SHAP values indicate dissimilarity.

                            Do not say whether the property is 'good' or 'bad'. Instead, explain how the model interpreted the 
                            feature similarities or differences that affected the score."""

                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"""
                        The model assigned the candidate property at {candidate_address} a score of {score:.2f} when comparing it to the subject at {subject_address}.

                        Features that made the candidate more similar:
                        {', '.join([f'{f} ({v:.2f})' for f, v in pos_feats]) or 'None'}

                        Features that made the candidate less similar:
                        {', '.join([f'{f} ({v:.2f})' for f, v in neg_feats]) or 'None'}

                        Based only on the features and their impact, describe why the model produced this score IN NONE TECHNICAL TERMS. Focus on the similarity or difference in attributes.
                        There is no need to mention "similarity score" or anything like that. Simply explain why the model selected this property as if you were a real
                        appraiser.  Think sentences like "The two properties have similar GLA, bedrooms, and bathrooms.  They differ a bit in lot size but ultimately this 
                        was not enough to outweigh their similarities." Do not mention specific SHAP score or anything just focus on the property features and similarities.
                        """
                    )
                }
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error getting GPT explanation: {e}]"

def gpt_explanation(score, pos_feats, neg_feats, candidate_address, subject_address, row):
    def enrich(features):
        return ', '.join(
            f"{f} = {row.get(f, 'N/A')} (SHAP {v:+.2f})"
            for f, v in features
        )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a real estate appraisal assistant. Your job is to explain why a machine learning model ranked a candidate property as more or less comparable to a subject property.\n\n"
                        "The model uses feature differences (e.g., size difference, age difference) between the candidate and subject. Positive SHAP values mean the feature made the candidate more similar (better match), while negative SHAP values indicate dissimilarity.\n\n"
                        "Do not say whether the property is 'good' or 'bad'. Instead, explain how the model interpreted the feature similarities or differences that affected the score. Use both the actual feature values and their SHAP impact scores."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"The model gave the candidate property at {candidate_address} a score of {score:.2f} when comparing it to the subject at {subject_address}.\n\n"
                        f"These features made the candidate more similar:\n{enrich(pos_feats) or 'None'}\n\n"
                        f"These features made the candidate less similar:\n{enrich(neg_feats) or 'None'}\n\n"
                        "Using the actual values and SHAP scores, explain in 1–2 sentences why the model ranked this candidate where it did."
                    )
                }
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error getting GPT explanation: {e}]"


# SHAP wrapper  
def model_predict(X_df):
    dmatrix = xgb.DMatrix(X_df)
    return model.predict(dmatrix)

explainer = shap.Explainer(model_predict, df[feature_cols])

# Main loop 
results = []
for order_id, group in tqdm(df.groupby("orderID"), desc="Generating GPT Explanations"):
    group = group.copy()
    group[feature_cols] = group[feature_cols].astype(float)
    dmatrix = xgb.DMatrix(group[feature_cols])
    group["score"] = model.predict(dmatrix)
    group["rank"] = group["score"].rank(method="first", ascending=False)

    top3 = group.sort_values("score", ascending=False).head(3)

    for _, row in top3.iterrows():
        row_df = row[feature_cols].to_frame().T.astype(float)
        try:
            shap_vals = explainer(row_df)
        except Exception as e:
            print(f"[SHAP Error] orderID={order_id}: {e}")
            continue

        shap_items = list(zip(row_df.columns, shap_vals.values[0]))
        positive_factors = [(f, v) for f, v in shap_items if v > 0]
        negative_factors = [(f, v) for f, v in shap_items if v < 0]

        extra = find_raw_values(order_id, row["candidate_address"])
        enriched_row = row.to_dict() | extra | {"orderID": order_id}

        explanation = gpt_explanation(
            row['score'], positive_factors[:3], negative_factors[:3],
            row["candidate_address"], row["subject_address"], enriched_row
        )

        enriched_row["explanation"] = explanation
        results.append(enriched_row)

# Final output 
top3_df = pd.DataFrame(results)
top3_df = top3_df.sort_values(by=["orderID", "score"], ascending=[True, False])
top3_df.to_csv("top3_gpt_explanations.csv", index=False)
print("\nSaved top3_gpt_explanations.csv")

# Analysis 
print("\n[Results Analysis]")
print("Total top-3 rows:", len(top3_df))
print("How many are labeled comps (is_comp = 1)?", top3_df["is_comp"].sum())
print("Top-3 Precision:", top3_df["is_comp"].mean())
print(top3_df["is_comp"].value_counts())

false_positives = top3_df[top3_df["is_comp"] == 0][["orderID", "candidate_address"]]
print("\nFalse Positives (Top-3 predicted but not comps):")
print(false_positives.to_string(index=False))
