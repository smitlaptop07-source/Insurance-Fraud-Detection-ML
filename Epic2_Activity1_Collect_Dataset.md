# Epic 2 - Activity 1 : Collect the Dataset

## Dataset Source
Dataset is collected from Kaggle (open source platform).

## Dataset Link
https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data

## Dataset Details
- Format: CSV (Comma Separated Values)
- Contains historical auto insurance claims data
- Target variable: `fraud_reported` (Y = Fraud, N = Not Fraud)

## Key Features in Dataset
- policy_number, policy_bind_date, policy_state
- insured_age, insured_sex, insured_education_level
- incident_type, collision_type, incident_severity
- authorities_contacted, incident_hour_of_the_day
- number_of_vehicles_involved, bodily_injuries
- total_claim_amount, injury_claim, property_claim
- fraud_reported (Target Variable)

## Activity 1.1 : Importing the Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

plt.style.use('fivethirtyeight')
```

## Activity 1.2 : Read the Dataset

```python
# Read the dataset
df = pd.read_csv('insurance_claims.csv')

# View first 5 rows
df.head()

# Check shape
print(df.shape)

# Check null values
print(df.isna().any())
print(df.isna().sum())
```

### Observations
- Dataset is read using pandas read_csv() function
- Checked for null values using df.isna().any()
- No null values found in the dataset
- Missing value handling step can be skipped
