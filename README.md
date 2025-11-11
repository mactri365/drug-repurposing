# AI-Based Drug Repurposing Application

This application uses machine learning to predict the effectiveness of existing drugs for different medical conditions based on historical data, reviews, and other factors.

## Prerequisites

Python 3.7 or higher installed on your system.

## Setup Instructions

1. Clone or download the project files to your local machine
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure all model files are present:
   - `drug_model.pkl`
   - `le_drug.pkl`
   - `le_condition.pkl`
   - `tfidf_vectorizer.pkl`

## How to Run

### Option 1: Interactive Web Application
1. If you don't have the model files, run the training script first:
   ```bash
   python drug_repurposing.py
   ```

2. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

3. Once the application is running, your browser should open automatically to the app interface. If not, look for the local URL in the console output (typically http://localhost:8501).

### Option 2: Drug Repurposing Analysis
Run the analysis script to identify potential drug repurposing candidates:
```bash
python drugrepurposing2.py
```

This script will:
- Train a model on the drug dataset
- Identify drugs that are effective for multiple conditions
- Generate insights for potential drug repurposing
- Save results to `repurposed_drugs.csv`
- Create visualizations of top repurposed drugs

### Option 3: Interactive Drug Repurposing Web App
Run the second Streamlit application focused on drug repurposing insights:
```bash
streamlit run app2.py
```

This application will:
- Train a model on the drug dataset in real-time
- Show potential drug repurposing candidates
- Allow you to search for specific drugs and see their repurposing possibilities
- Display visualizations of top repurposed drugs
- Provide insights on how existing drugs can be used for new conditions

## Features

### Web Application
- Predict drug effectiveness ratings (1-10 scale)
- Enter drug name, medical condition, and patient review
- Adjust helpful votes slider
- Visual feedback based on prediction results
- Sample of top predictions

### Analysis Script
- Comprehensive drug repurposing insights
- Identification of drugs effective for multiple conditions
- Export of potential repurposing candidates
- Visualizations of top repurposed drugs

## Data Source

The application uses the drugsCom dataset (train and test) to train the machine learning model.

## Technical Details

- Frontend: Streamlit for web interface
- ML Model: Linear Regression with TF-IDF text vectorization
- Preprocessing: Label encoding for categorical variables
- Features: Drug name, condition, review text, number of helpful votes, review length