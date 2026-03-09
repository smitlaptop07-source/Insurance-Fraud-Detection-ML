import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')

print("="*60)
print("   INSURANCE FRAUD DETECTION - DATA ANALYSIS")
print("="*60)

# ===================== LOAD DATA =====================
df = pd.read_csv('insurance_claims.csv')
print("\n[1] Dataset loaded successfully!")
print("Shape:", df.shape)

# ===================== UNIVARIATE ANALYSIS =====================
print("\n[3] Univariate Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Univariate Analysis - Insurance Fraud Detection', fontsize=16)

df['fraud_reported'].value_counts().plot(kind='bar', ax=axes[0,0], color=['#2ecc71','#e74c3c'])
axes[0,0].set_title('Fraud vs Non-Fraud Claims')
axes[0,0].tick_params(axis='x', rotation=0)

axes[0,1].hist(df['age'], bins=30, color='#3498db', edgecolor='white')
axes[0,1].set_title('Age Distribution')

axes[1,0].hist(df['total_claim_amount'], bins=40, color='#e67e22', edgecolor='white')
axes[1,0].set_title('Total Claim Amount Distribution')

df['incident_severity'].value_counts().plot(kind='bar', ax=axes[1,1], color='#9b59b6')
axes[1,1].set_title('Incident Severity Distribution')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('univariate_analysis.png', dpi=100, bbox_inches='tight')
plt.show()
print("Univariate analysis saved!")

# ===================== MULTIVARIATE ANALYSIS =====================
print("\n[4] Multivariate Analysis (Heatmap)...")
numeric_df = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=100, bbox_inches='tight')
plt.show()
print("Heatmap saved!")

# ===================== DATA PREPROCESSING =====================
print("\n[5] Data Preprocessing...")

# Drop unnamed column
if '_c39' in df.columns:
    df.drop('_c39', axis=1, inplace=True)

# Replace ? with NaN in string columns only
obj_cols = df.select_dtypes(include='object').columns
for col in obj_cols:
    df[col] = df[col].replace('?', np.nan)
    df[col].fillna(df[col].mode()[0], inplace=True)

# Fill numeric NaN with median
num_cols = df.select_dtypes(include=['float64','int64']).columns
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

print("Missing values remaining:", df.isnull().sum().sum())
print("Shape after handling nulls:", df.shape)

# Drop unnecessary columns
drop_cols = ['policy_number', 'policy_bind_date', 'insured_zip',
             'incident_location', 'auto_make', 'auto_model',
             'months_as_customer', 'injury_claim', 'property_claim', 'vehicle_claim']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
print("Shape after dropping features:", df.shape)

# ===================== ENCODING =====================
print("\n[6] Encoding categorical features...")
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if 'fraud_reported' in categorical_cols:
    categorical_cols.remove('fraud_reported')
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))
df['fraud_reported'] = le.fit_transform(df['fraud_reported'])
print("Encoding complete! Shape:", df.shape)

# ===================== FEATURES & TARGET =====================
X = df.drop('fraud_reported', axis=1)
y = df['fraud_reported']
print("\nFeatures shape:", X.shape)
print("Target distribution:\n", y.value_counts())

# ===================== SCALING =====================
print("\n[7] Scaling features...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Train size:", X_train_scaled.shape)
print("Test size:", X_test_scaled.shape)

# ===================== SAVE =====================
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump((X_train_scaled, X_test_scaled, y_train, y_test), open('data.pkl', 'wb'))
print("\nScaler saved!")
print("Data saved!")
print("\n[EDA COMPLETE] Run models.py next!")
