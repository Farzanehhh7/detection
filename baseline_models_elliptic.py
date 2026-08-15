import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, precision_score, recall_score
)


FEATURES_PATH = "datasets/elliptic_txs_features.csv"
CLASSES_PATH = "datasets/elliptic_txs_classes.csv"


SVM_TRAIN_SAMPLE = None


feat_cols = ["txId", "time_step"] + [f"feat_{i}" for i in range(165)]
df_feat = pd.read_csv(FEATURES_PATH, header=None, names=feat_cols)

df_class = pd.read_csv(CLASSES_PATH)
df_class.columns = ["txId", "class"]

print(f"Total number of transactions (nodes): {len(df_feat)}")


df = df_feat.merge(df_class, on="txId", how="left")

df = df[df["class"] != "unknown"].copy()

df["label"] = df["class"].map({"1": 1, "2": 0})

print(f"Number of labeled samples: {len(df)}")
print(f"illicit: {(df['label'] == 1).sum()}   licit: {(df['label'] == 0).sum()}")


train_df = df[df["time_step"] <= 34]
test_df = df[df["time_step"] > 34]

feature_cols = [c for c in df.columns if c.startswith("feat_")]

X_train, y_train = train_df[feature_cols], train_df["label"]
X_test, y_test = test_df[feature_cols], test_df["label"]

print(f"\nTrain: {len(X_train)} samples ({y_train.sum()} illicit)")
print(f"Test:  {len(X_test)} samples ({y_test.sum()} illicit)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


results = []


def evaluate(name, y_true, y_pred, y_prob):
    row = {
        "model": name,
        "AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "F1_illicit": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    print(f"\n--- {name} ---")
    for k, v in row.items():
        if k != "model":
            print(f"{k:12s}: {v:.4f}")
    results.append(row)



lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
evaluate("Logistic Regression", y_test, lr.predict(X_test_scaled), lr.predict_proba(X_test_scaled)[:, 1])


rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
evaluate("Random Forest", y_test, rf.predict(X_test), rf.predict_proba(X_test)[:, 1])


n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = n_neg / n_pos
print(f"\n[XGBoost] Calculated scale_pos_weight (Nneg/Npos): {scale_pos_weight:.2f}")

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
)
xgb.fit(X_train, y_train)
evaluate("XGBoost", y_test, xgb.predict(X_test), xgb.predict_proba(X_test)[:, 1])


if SVM_TRAIN_SAMPLE:
    idx = X_train.sample(SVM_TRAIN_SAMPLE, random_state=42).index
    X_train_svm = pd.DataFrame(X_train_scaled, index=X_train.index).loc[idx]
    y_train_svm = y_train.loc[idx]
else:
    X_train_svm, y_train_svm = X_train_scaled, y_train

svm = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)
svm.fit(X_train_svm, y_train_svm)
evaluate("SVM", y_test, svm.predict(X_test_scaled), svm.predict_proba(X_test_scaled)[:, 1])


results_df = pd.DataFrame(results).set_index("model")
print("\n\n=== Final Baseline Comparison Table ===")
print(results_df.round(4))
print("\nFor comparison: Weber's paper F1 with GCN on the same time split = 0.705")

results_df.to_csv("baseline_results.csv")
print("\nResults successfully saved to baseline_results.csv")