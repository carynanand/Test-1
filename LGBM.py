# pip install lightgbm scikit-learn imbalanced-learn shap

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,        # L1 — handles irrelevant features
    reg_lambda=1.0,       # L2 — prevents overfitting
    class_weight="balanced",  # critical: pathogenic variants are rare
    n_jobs=-1,
    random_state=42,
)

# Use StratifiedKFold — class imbalance is severe in germline data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
