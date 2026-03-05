import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("   INSURANCE FRAUD DETECTION - MODEL BUILDING")
print("="*60)

# ===================== LOAD DATA =====================
X_train, X_test, y_train, y_test = pickle.load(open('data.pkl', 'rb'))
print("\nData loaded successfully!")
print("Train:", X_train.shape, "| Test:", X_test.shape)

# ===================== DEFINE MODELS =====================
models = {
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes":         GaussianNB(),
    "SVM":                 SVC(kernel='rbf', probability=True, random_state=42)
}

# ===================== TRAIN & EVALUATE =====================
results = {}

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, model.predict_proba(X_test)[:,1]
                               if hasattr(model, 'predict_proba')
                               else model.decision_function(X_test))

    results[name] = {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'roc_auc': roc_auc
    }

    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Test Accuracy  : {test_acc:.4f}")
    print(f"ROC AUC Score  : {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
          target_names=['Legitimate', 'Fraud']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate','Fraud'],
                yticklabels=['Legitimate','Fraud'])
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'cm_{name.replace(" ","_")}.png', dpi=80)
    plt.close()

# ===================== COMPARE MODELS =====================
print("\n" + "="*60)
print("   MODEL COMPARISON")
print("="*60)

comparison = pd.DataFrame({
    'Model': list(results.keys()),
    'Train Accuracy': [results[m]['train_acc'] for m in results],
    'Test Accuracy':  [results[m]['test_acc']  for m in results],
    'ROC AUC':        [results[m]['roc_auc']   for m in results]
}).sort_values('Test Accuracy', ascending=False)

print(comparison.to_string(index=False))

# Bar chart comparison
plt.figure(figsize=(12, 5))
x = np.arange(len(comparison))
width = 0.3
plt.bar(x - width, comparison['Train Accuracy'], width, label='Train Acc', color='#3498db')
plt.bar(x,         comparison['Test Accuracy'],  width, label='Test Acc',  color='#2ecc71')
plt.bar(x + width, comparison['ROC AUC'],        width, label='ROC AUC',  color='#e74c3c')
plt.xticks(x, comparison['Model'], rotation=20, ha='right')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=100, bbox_inches='tight')
plt.show()
print("\nComparison chart saved as model_comparison.png")

# ===================== SAVE BEST MODEL =====================
best_name = comparison.iloc[0]['Model']
best_model = results[best_name]['model']
pickle.dump(best_model, open('model.pkl', 'wb'))

print(f"\n✅ Best Model: {best_name}")
print(f"   Test Accuracy : {comparison.iloc[0]['Test Accuracy']:.4f}")
print(f"   ROC AUC Score : {comparison.iloc[0]['ROC AUC']:.4f}")
print("\nModel saved as model.pkl")
print("\n[MODELS COMPLETE] Run app.py to launch the web app!")
