import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier, LGBMRegressor
import mlflow
import mlflow.sklearn
import shap
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parents[1] / 'data' / 'synthetic' / 'features.csv'
OUTD = Path(__file__).resolve().parents[1] / 'training' / 'artifacts'
OUTD.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(DATA)
    group_col = 'batch_id' if 'batch_id' in df.columns else 'kettle' if 'kettle' in df.columns else None
    if group_col:
        df = df.sort_values(group_col)

    y_cls = df['pass_flag'].astype(int)
    y_reg = df['viscosity']

    X = df.drop(columns=['pass_flag','viscosity','free_hcho','moisture','dextrin',
                         'sec_cut_2h','sec_cut_24h','hardness','penetration'])

    num_cols = X.columns.tolist()
    tscv = TimeSeriesSplit(n_splits=5)
    mlflow.set_tracking_uri(OUTD.as_uri())

    cls_scores = []
    for train_idx, test_idx in tscv.split(X, y_cls):
        pre = ColumnTransformer([('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                                                   ('sc', StandardScaler())]), num_cols)], remainder='drop')
        clf = Pipeline([('pre', pre),
                        ('lgbm', LGBMClassifier(num_leaves=63, learning_rate=0.05, n_estimators=600,
                                                feature_fraction=0.8, bagging_fraction=0.8, min_data_in_leaf=40))])
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_cls.iloc[train_idx], y_cls.iloc[test_idx]
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, proba)
        brier = brier_score_loss(y_test, proba)
        cls_scores.append((auc, brier))

    avg_auc = float(np.mean([s[0] for s in cls_scores]))
    avg_brier = float(np.mean([s[1] for s in cls_scores]))

    # log classification
    with mlflow.start_run(run_name='classification'):
        mlflow.log_params(clf.named_steps['lgbm'].get_params())
        mlflow.log_metric('auc', avg_auc)
        mlflow.log_metric('brier', avg_brier)
        mlflow.sklearn.log_model(clf, 'model')
        pre = clf.named_steps['pre']
        X_trans = pd.DataFrame(pre.transform(X), columns=num_cols)
        explainer = shap.TreeExplainer(clf.named_steps['lgbm'])
        shap_values = explainer.shap_values(X_trans)
        shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values,
                          X_trans, plot_type='bar', show=False)
        cls_shap = OUTD / 'cls_shap.png'
        plt.tight_layout()
        plt.savefig(cls_shap)
        plt.close()
        mlflow.log_artifact(cls_shap)

    # Regression for viscosity
    XR = X.copy()
    reg_scores = []
    for train_idx, test_idx in tscv.split(XR, y_reg):
        pre = ColumnTransformer([('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                                                   ('sc', StandardScaler())]), num_cols)], remainder='drop')
        reg = Pipeline([('pre', pre),
                        ('lgbm', LGBMRegressor(num_leaves=63, learning_rate=0.05, n_estimators=800,
                                               feature_fraction=0.8, bagging_fraction=0.8, min_data_in_leaf=40))])
        X_train, X_test = XR.iloc[train_idx], XR.iloc[test_idx]
        y_train, y_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
        reg.fit(X_train, y_train)
        pred = reg.predict(X_test)
        reg_scores.append(mean_absolute_error(y_test, pred))

    avg_mae = float(np.mean(reg_scores))
    with mlflow.start_run(run_name='regression'):
        mlflow.log_params(reg.named_steps['lgbm'].get_params())
        mlflow.log_metric('mae', avg_mae)
        mlflow.sklearn.log_model(reg, 'model')
        pre = reg.named_steps['pre']
        X_trans = pd.DataFrame(pre.transform(XR), columns=num_cols)
        explainer = shap.TreeExplainer(reg.named_steps['lgbm'])
        shap_values = explainer.shap_values(X_trans)
        shap.summary_plot(shap_values, X_trans, plot_type='bar', show=False)
        reg_shap = OUTD / 'reg_shap.png'
        plt.tight_layout()
        plt.savefig(reg_shap)
        plt.close()
        mlflow.log_artifact(reg_shap)

    print('Logged runs to', OUTD)

if __name__ == '__main__':
    main()
