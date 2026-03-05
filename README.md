# Insurance Fraud Detection Using Machine Learning

## Project Overview
A Machine Learning project that detects fraudulent insurance claims using Python and Flask.

## Project Structure
```
Insurance-Fraud-Detection-ML/
│
├── insurance_claims.csv        # Dataset (download from Kaggle)
├── eda.py                      # Step 1: Data Analysis & Preprocessing
├── models.py                   # Step 2: Train all 6 ML Models
├── performance_tuning.py       # Step 3: Hyperparameter Tuning
├── app.py                      # Step 4: Flask Web Application
├── templates/
│   └── index.html              # Web Interface
├── model.pkl                   # Saved best model (generated)
├── scaler.pkl                  # Saved scaler (generated)
└── data.pkl                    # Preprocessed data (generated)
```

## How to Run

### Step 1: Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn flask pickle5
```

### Step 2: Download Dataset
Download from Kaggle:
https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data

Save as `insurance_claims.csv` in the project folder.

### Step 3: Run EDA & Preprocessing
```bash
python eda.py
```

### Step 4: Train Models
```bash
python models.py
```

### Step 5: Performance Tuning (Optional)
```bash
python performance_tuning.py
```

### Step 6: Launch Web App
```bash
python app.py
```
Open browser: http://127.0.0.1:5000

## ML Models Used
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)

## Dataset
- Source: Kaggle
- Link: https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data

## Tech Stack
- Python 3.x | Pandas | NumPy | Scikit-learn | Matplotlib | Seaborn | Flask
