import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import os

SHUFFLE_LABELS = False

training_data_file = "training_data_with_feedback.csv" if os.path.exists("feedback_log.csv") and os.path.getsize("feedback_log.csv") > 0 else "training_data.csv"

# Load dataset
df = pd.read_csv(training_data_file)

print(f"Using training data: {training_data_file}")

if SHUFFLE_LABELS:
    print("Shuffling labels for sanity check...")
    df["is_comp"] = df.groupby("orderID")["is_comp"].transform(
        lambda x: np.random.permutation(x.values)
    )

# Define feature columns
feature_cols = [
    'bath_score_diff', 'full_baths_diff', 'half_baths_diff',
    'room_count_diff', 'bedrooms_diff', 'effective_age_diff',
    'subject_age_diff', 'lot_size_sf_diff', 'gla_diff',
    'abs_bath_score_diff', 'abs_full_bath_diff', 'abs_half_bath_diff',
    'abs_room_count_diff', 'abs_bedrooms_diff', 'abs_effective_age_diff',
    'abs_subject_age_diff', 'abs_lot_size_sf_diff', 'abs_gla_diff',
    'same_property_type', 'sold_recently', # 'distance_to_subject_km'
]

# Fill in label if not already present
df['label'] = df['is_comp']

# Train-test split
df_train, df_test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label']
)

# Sort for group creation
df_train = df_train.sort_values("orderID")
df_test = df_test.sort_values("orderID")

# Group by orderID for ranking
groups_train = df_train.groupby("orderID").size().to_list()

# Ensure numeric input (float) for DMatrix
X_train = df_train[feature_cols].astype(float)
y_train = df_train["label"]

dtrain = xgb.DMatrix(X_train, label=y_train)
dtrain.set_group(groups_train)

# Train ranking model
params = {
    'objective': 'rank:pairwise',
    'eval_metric': 'ndcg',
    'eta': 0.1,
    'max_depth': 6,
    'verbosity': 1
}

model = xgb.train(params, dtrain, num_boost_round=100)

# Evaluation
print("\nTop-K Evaluation by Appraisal:")

def evaluate_topk(df_group, k=3):
    df_group = df_group.copy()
    X = xgb.DMatrix(df_group[feature_cols].astype(float))
    df_group["score"] = model.predict(X)
    topk = df_group.sort_values("score", ascending=False).head(k)
    correct = topk["label"].sum()
    return pd.Series({"correct": correct, "total": k})

# Evaluate at K = 1, 3
for k in [1, 3]:
    results = df_test.groupby("orderID").apply(lambda g: evaluate_topk(g, k)).sum()
    precision = results["correct"] / results["total"]
    print(f"Top-{k} Precision: {precision:.3f}")

# Save the model
model.save_model("xgb_rank_model.json")
print("\nRanking model saved as xgb_rank_model.json")