# ML Property Ranking System

This project is an interactive AI-powered tool for evaluating and explaining comparable property rankings.

It uses:
- XGBoost for pairwise ranking
- SHAP for explainability
- GPT-3.5 for natural language explanations
- GPT-3.5 for property remark parsing for accurate data

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
- Uses ChatGPT-3.5 to parse public property remarks to improve data accuracy and completeness
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
- `gpt_extracted_features_appraisals.json`: Appraisal data parsed from public remarks
- `feature_engineered_appraisals_dataset.json`: Feature engineered appraisal data
- `geocoded_addresses.json`: Longitude and latitude data for each address in the dataset
- `training_data.csv`: Processed training dataset
- `training_data_with_feedback.csv`: Dataset with integrated user feedback
- `feedback_log.csv`: Log of submitted feedback
- `top3_gpt_explanations.csv`: Final output with model explanations

---

## Key Improvments this Iteration

- Parsing `public-remarks` data using ChatGPT-3.5 to get more reliable and accurate data. It also allowed for addition of new data like condition data and other property features
- Merging this new data into the final dataset, prioritizing public remarks data as ground truth
- This new data took the precision up to 98%, with condition difference being the most important new feature
- Removed some of the less important features according to SHAP feature analysis
## Thoughts on Future Improvements

- More data: Appraisers think about more than just things like "are there the same number of bedrooms" or "is the GLA similar". Things like trendy cabinetry or window sizes or general aesthetics are also important factors to consider for appraisers.
- System for adding new data points: The ability to add new appraisals, potential properties, etc.
- Better understanding of neighborhoods: Rather than just distance between coordinates, a system for better understanding whether or not two properties are in the same neighborhood, school district, etc.