import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier, LGBMRegressor
import joblib, json

DATA = Path(__file__).resolve().parents[1] / 'data' / 'synthetic' / 'features.csv'
OUTD = Path(__file__).resolve().parents[1] / 'training' / 'artifacts'
OUTD.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(DATA)
    y_cls = df['pass_flag'].astype(int)
    y_reg = df['viscosity']

    X = df.drop(columns=['pass_flag','viscosity','free_hcho','moisture','dextrin',
                         'sec_cut_2h','sec_cut_24h','hardness','penetration'])

    num_cols = X.columns.tolist()
    pre = ColumnTransformer([('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                                               ('sc', StandardScaler())]), num_cols)], remainder='drop')

    X_train, X_test, y_train, y_test = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    clf = Pipeline([('pre', pre),
                    ('lgbm', LGBMClassifier(num_leaves=63, learning_rate=0.05, n_estimators=600,
                                            feature_fraction=0.8, bagging_fraction=0.8, min_data_in_leaf=40))])
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    joblib.dump(clf, OUTD/'cls_pass.pkl')
    with open(OUTD/'cls_metrics.json','w') as f:
        json.dump({'auc':float(auc), 'brier':float(brier)}, f, indent=2)

    # Regression for viscosity
    yR = df['viscosity']; XR = X.copy()
    X_train, X_test, y_train, y_test = train_test_split(XR, yR, test_size=0.2, random_state=42)
    reg = Pipeline([('pre', pre),
                    ('lgbm', LGBMRegressor(num_leaves=63, learning_rate=0.05, n_estimators=800,
                                           feature_fraction=0.8, bagging_fraction=0.8, min_data_in_leaf=40))])
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    joblib.dump(reg, OUTD/'reg_viscosity.pkl')
    with open(OUTD/'reg_metrics.json','w') as f:
        json.dump({'mae':float(mae)}, f, indent=2)

    print('Saved models & metrics ->', OUTD)

if __name__ == '__main__':
    main()
