# AIDA (AI-Driven Churn Analytics)

AIDA is an intelligent system for predicting customer churn and providing retention strategies using Machine Learning and AI.

## Features
- **Churn Prediction**: Uses XGBoost + SMOTE for accurate predictions.
- **Risk Analysis**: Categorizes customers into High, Medium, or Low risk.
- **AI Recommendations**: Generates personalized retention strategies using Gemini AI.
- **Interactive Dashboard**: Visualizes data and model performance.
- **AutoML Training**: Automatically retrains models with new data.

## Installation
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Set up `.env` file with `DATABASE_URL` and `GEMINI_API_KEY`.
4. Run the server: `python main.py`.

## Usage
- Open `http://127.0.0.1:8000` in your browser.
- Use the dashboard to analyze churn and view predictions.
