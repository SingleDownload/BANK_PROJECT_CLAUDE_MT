# 🏦 Universal Bank — Loan Intelligence Dashboard

A professional analytics and ML dashboard built with Streamlit for the Universal Bank marketing team. Covers the full analytics journey from descriptive to prescriptive, plus live predictions on new customer data.

## 📊 Dashboard Sections

| Section | Description |
|---|---|
| Executive Overview | KPIs, class imbalance, model comparison at a glance |
| Descriptive Analytics | Distributions, correlations, summary statistics |
| Diagnostic Analytics | Why do customers accept loans? Segment deep-dives |
| Predictive Models | DT / RF / GBT — accuracy table, ROC curve, confusion matrices |
| Prescriptive Analytics | Budget allocation, ideal customer profile, campaign recommendations |
| Predict New Customers | Upload CSV → download predictions with probability scores |

## 🚀 Deployment on Streamlit Community Cloud

1. **Fork / push this repo to GitHub**
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**
3. Select your repo, branch `main`, and set **Main file path** to `app.py`
4. Click **Deploy** — done!

## 🗂️ Project Structure

```
├── app.py                  # Main Streamlit application
├── UniversalBank.csv       # Training dataset (5,000 customers)
├── test_data_sample.csv    # Sample test file for the Predict page
├── requirements.txt        # Python dependencies
└── README.md
```

## 💻 Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🧠 Models Used

- **Decision Tree** — interpretable, balanced class weights, max_depth=8
- **Random Forest** — 150 estimators, balanced class weights
- **Gradient Boosted Tree** — 150 estimators, learning_rate=0.1, best overall AUC

## 📦 Dependencies

All listed in `requirements.txt`. No `imbalanced-learn` required — class imbalance is handled via `class_weight='balanced'` in scikit-learn directly.

## 📁 Test Data

Upload `test_data_sample.csv` on the **Predict New Customers** page to see live predictions. The file contains 200 synthetic customers with the same schema as the training data (no `Personal Loan` column).
