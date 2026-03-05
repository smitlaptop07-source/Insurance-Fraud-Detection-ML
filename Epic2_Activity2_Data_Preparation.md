# Epic 2 - Activity 2 : Data Preparation

## Steps in Data Preparation

### Step 1 : Handle Missing Values
```python
# Check missing values
print(df.isnull().sum())

# Replace '?' with NaN
df = df.replace('?', np.nan)

# Drop rows with missing values (if any)
df.dropna(inplace=True)
```

### Step 2 : Drop Unnecessary Columns
```python
# Drop columns not useful for prediction
df.drop(['policy_number', 'policy_bind_date', 'insured_zip',
         'incident_location', 'auto_make', 'auto_model'], axis=1, inplace=True)
```

### Step 3 : Encode Categorical Variables
```python
# Label Encoding for categorical columns
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
```

### Step 4 : Define Features and Target
```python
# X = features, y = target
X = df.drop('fraud_reported', axis=1)
y = df['fraud_reported']
```

### Step 5 : Handle Imbalanced Data using SMOTE
```python
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print("After SMOTE:", X_res.shape, y_res.shape)
```

### Step 6 : Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)
```

## Summary
- Missing values handled
- Unnecessary columns removed
- Categorical variables encoded using Label Encoding
- Imbalanced data handled using SMOTE
- Data split into 80% train and 20% test
