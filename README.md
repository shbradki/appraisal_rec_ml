# ML Property Ranking System

This project is an interactive AI-powered tool for evaluating and explaining comparable property rankings.

It uses:
- XGBoost for pairwise ranking
- SHAP for explainability
- GPT-3.5 for natural language explanations

---

## Getting Started

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Set your OpenAI API key**

Make sure your OpenAI API key is exported as an environment variable:

```bash
export OPENAI_API_KEY=your-key-here
```

3. **Run the app**

Launch the Streamlit interface:

```bash
streamlit run app.py
```

---

## How It Works

- Reads a JSON dataset of appraisals and candidate properties
- Cleans/parses the necessary appraisal data
- Runs geocoding for all addresses if needed
- Performs feature engineering on each candidate vs. subject
- Trains a ranking model to score candidate comparables
- Uses SHAP to compute feature-level impact for each of the top-3 ranked comps
- Uses GPT-3.5 to explain the rankings in natural language

---

## Feedback Loop

Users can provide feedback on poor comp predictions directly in the UI:
- Feedback is logged and integrated into the next training cycle
- Bad comps are dropped entirely
- Over time, the model learns from user guidance and improves

---

## Files

- `appraisals_dataset.json`: Input data
- `cleaned_appraisals_dataset.json`: Cleaned/parsed appraisal data
- `feature_engineered_appraisals_dataset.json`: Feature engineered appraisal data
- `geocoded_addresses.json`: Longitude and latitude data for each address in the dataset
- `training_data.csv`: Processed training dataset
- `training_data_with_feedback.csv`: Dataset with integrated user feedback
- `feedback_log.csv`: Log of submitted feedback
- `top3_gpt_explanations.csv`: Final output with model explanations

---

