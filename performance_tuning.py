import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("   INSURANCE FRAUD DETECTION - PERFORMANCE TUNING")
print("="*60)

# ===================== LOAD DATA =====================
X_train, X_test, y_train, y_test = pickle.load(open('data.pkl', 'rb'))
print("\nData loaded!")

# ===================== CROSS VALIDATION =====================
print("\n[1] Cross Validation (5-Fold) on Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')

print("CV Scores:", [round(s, 4) for s in cv_scores])
print(f"Mean CV Score : {cv_scores.mean():.4f}")
print(f"Std Deviation : {cv_scores.std():.4f}")

plt.figure(figsize=(8,4))
plt.plot(range(1,6), cv_scores, marker='o', color='#3498db', linewidth=2)
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', label=f'Mean={cv_scores.mean():.4f}')
plt.title('5-Fold Cross Validation Scores')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('cross_validation.png', dpi=100)
plt.show()
print("Cross validation chart saved!")

# ===================== BEFORE TUNING =====================
print("\n[2] Accuracy BEFORE Hyperparameter Tuning...")
rf.fit(X_train, y_train)
before_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"Test Accuracy (before tuning): {before_acc:.4f}")

# ===================== HYPERPARAMETER TUNING =====================
print("\n[3] Hyperparameter Tuning with GridSearchCV...")
print("This may take a few minutes...")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)
print(f"Best CV Score  : {grid.best_score_:.4f}")

# ===================== AFTER TUNING =====================
print("\n[4] Accuracy AFTER Hyperparameter Tuning...")
best_model = grid.best_estimator_
after_acc = accuracy_score(y_test, best_model.predict(X_test))
print(f"Test Accuracy (after tuning): {after_acc:.4f}")

# Compare before vs after
print("\n--- Comparison ---")
print(f"Before Tuning : {before_acc:.4f}")
print(f"After Tuning  : {after_acc:.4f}")
print(f"Improvement   : {(after_acc - before_acc):.4f}")

# ===================== SAVE TUNED MODEL =====================
pickle.dump(best_model, open('model.pkl', 'wb'))
print("\n✅ Tuned model saved as model.pkl")
print("\n[TUNING COMPLETE] Run app.py to launch the web app!")
