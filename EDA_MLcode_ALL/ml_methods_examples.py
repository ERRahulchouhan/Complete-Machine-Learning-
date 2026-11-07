import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import zscore
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    SMOTE = None
    ImbPipeline = None
from joblib import dump, load


# -----------------------------------------------------------------------------
# 1. Read the data
# -----------------------------------------------------------------------------

def read_data_examples():
    df_csv = pd.read_csv("data.csv")
    df_excel = pd.read_excel("data.xlsx")
    df_json = pd.read_json("data.json")
    df_parquet = pd.read_parquet("data.parquet")
    return {"csv": df_csv, "excel": df_excel, "json": df_json, "parquet": df_parquet}


# -----------------------------------------------------------------------------
# 2. Quick checks and EDA
# -----------------------------------------------------------------------------

def quick_eda(df):
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Dtypes:\n", df.dtypes)
    print("Missing values:\n", df.isnull().sum())
    print("Summary:\n", df.describe(include="all").T)
    print("Duplicate rows:", df.duplicated().sum())
    print(df.head())

    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns

    for col in num_cols[:3]:
        print(f"{col} skew = {df[col].skew():.3f}, kurtosis = {df[col].kurt():.3f}")
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()

    for col in cat_cols[:3]:
        sns.countplot(data=df, x=col)
        plt.title(f"Frequency of {col}")
        plt.show()


# -----------------------------------------------------------------------------
# 3. Missing values analysis and treatment
# -----------------------------------------------------------------------------

def missing_value_examples(df):
    print(df.isnull().sum())
    print(df.isna().mean() * 100)

    df_numeric = df.copy()
    df_numeric["age"] = df_numeric["age"].fillna(df_numeric["age"].median())
    df_numeric["city"] = df_numeric["city"].fillna(df_numeric["city"].mode()[0])
    df_numeric = df_numeric.ffill().bfill()

    simple_imputer = SimpleImputer(strategy="median")
    num_cols = df.select_dtypes(include=np.number).columns
    df_simple = pd.DataFrame(simple_imputer.fit_transform(df[num_cols]), columns=num_cols)

    knn_imputer = KNNImputer(n_neighbors=2)
    df_knn = pd.DataFrame(knn_imputer.fit_transform(df[num_cols]), columns=num_cols)

    imputer_mean = SimpleImputer(strategy="mean")
    df_mean = pd.DataFrame(imputer_mean.fit_transform(df_numeric.select_dtypes(include=np.number)))
    return {"manual_fill": df_numeric, "simple_imputer": df_simple, "knn_imputer": df_knn, "mean_imputer": df_mean}


# -----------------------------------------------------------------------------
# 4. Data cleaning / quality checks
# -----------------------------------------------------------------------------

def data_cleaning_examples(df):
    df = df.drop_duplicates()
    df["city"] = df["city"].astype(str).str.strip().str.lower()
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.rename(columns={"old_col": "new_col"})
    mask = df["age"].between(0, 110)
    df = df.loc[mask].copy()
    return df


# -----------------------------------------------------------------------------
# 5. Categorical analysis and encoding
# -----------------------------------------------------------------------------

def categorical_analysis(df):
    print(df["city"].value_counts())
    print(pd.crosstab(df["city"], df["target"]))

    freq = df["city"].value_counts(normalize=True)
    df["city_frequency"] = df["city"].map(freq)
    return df


def categorical_encoding_examples(df):
    X = df[["city", "gender", "size"]].copy()

    # 1) map() / replace() for manual category-to-number conversion
    city_map = {"Paris": 1, "London": 2, "Tokyo": 3, "New York": 4}
    X["city_map"] = X["city"].map(city_map)
    X["city_replace"] = X["city"].replace(city_map)

    # 2) category codes
    X["city_cat_codes"] = X["city"].astype("category").cat.codes

    # 3) factorize: returns codes and uniques
    X["city_factorized"], uniques = pd.factorize(X["city"])

    # 4) frequency encoding
    freq = X["city"].value_counts(normalize=True)
    X["city_frequency"] = X["city"].map(freq)

    # 5) label encoding
    label_encoder = LabelEncoder()
    X["city_label"] = label_encoder.fit_transform(X["city"])

    # 6) one-hot encoding
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    city_ohe = ohe.fit_transform(X[["city"]])
    city_ohe_df = pd.DataFrame(city_ohe, columns=ohe.get_feature_names_out(["city"]))

    # 7) ordinal encoding
    ord_enc = OrdinalEncoder(categories=[["small", "medium", "large"]])
    X["size_ord"] = ord_enc.fit_transform(X[["size"]])

    # 8) dummy encoding
    dummy_df = pd.get_dummies(df[["city", "gender"]], columns=["city"], drop_first=True)
    return X, city_ohe_df, dummy_df


# -----------------------------------------------------------------------------
# 6. Numerical analysis and outlier detection
# -----------------------------------------------------------------------------

def numerical_analysis_and_outliers(df):
    age = df["age"].dropna()
    print(age.mean(), age.median(), age.std(), age.skew(), age.kurt())
    stats.probplot(age, dist="norm", plot=plt)
    plt.show()

    q1 = age.quantile(0.25)
    q3 = age.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = df[(df["age"] < lower) | (df["age"] > upper)]
    print("Outlier count:", len(outliers))

    z = np.abs(zscore(age))
    outlier_mask = z > 3
    print("Z-score outliers:", np.sum(outlier_mask))

    df["age_clipped"] = df["age"].clip(lower=lower, upper=upper)
    return df


# -----------------------------------------------------------------------------
# 7. Feature scaling and transformations
# -----------------------------------------------------------------------------

def feature_scaling_examples(df):
    num_cols = df.select_dtypes(include=np.number).columns
    X_num = df[num_cols]

    scaler_std = StandardScaler().fit(X_num)
    scaled_std = scaler_std.transform(X_num)

    scaler_mm = MinMaxScaler().fit(X_num)
    scaled_mm = scaler_mm.transform(X_num)

    scaler_rob = RobustScaler().fit(X_num)
    scaled_rob = scaler_rob.transform(X_num)

    power = PowerTransformer(method="yeo-johnson")
    scaled_power = power.fit_transform(X_num)

    return {
        "standard": scaled_std,
        "minmax": scaled_mm,
        "robust": scaled_rob,
        "power": scaled_power,
    }


# -----------------------------------------------------------------------------
# 8. Feature engineering and transformation
# -----------------------------------------------------------------------------

def feature_engineering_examples(df):
    df = df.copy()
    df["log_income"] = np.log1p(df["salary"])
    df["age_sq"] = np.square(df["age"])
    df["age_bin"] = pd.cut(df["age"], bins=[0, 18, 35, 50, 65, 100], labels=["0-18", "19-35", "36-50", "51-65", "65+"])
    df["date_year"] = pd.to_datetime(df["date"]).dt.year
    df["date_month"] = pd.to_datetime(df["date"]).dt.month

    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(df[["age", "salary"]])
    return df, poly_features


# -----------------------------------------------------------------------------
# 9. Feature selection and dimensionality reduction
# -----------------------------------------------------------------------------

def feature_selection_examples(X, y):
    selector = SelectKBest(score_func=mutual_info_classif, k=3)
    X_sel = selector.fit_transform(X, y)

    rfecv = RFE(estimator=LogisticRegression(max_iter=500), n_features_to_select=3)
    X_rfe = rfecv.fit_transform(X, y)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    return X_sel, X_rfe, X_pca


# -----------------------------------------------------------------------------
# 10. Train/test split and cross-validation
# -----------------------------------------------------------------------------

def split_and_cross_validate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ts_split = TimeSeriesSplit(n_splits=5)

    model = LogisticRegression(max_iter=500)
    scores = cross_val_score(model, X_train, y_train, cv=skfold)
    print("CV scores:", scores)
    return X_train, X_test, y_train, y_test


# -----------------------------------------------------------------------------
# 11. Regression algorithms
# -----------------------------------------------------------------------------

def regression_examples(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.01),
        "elastic_net": ElasticNet(alpha=0.01, l1_ratio=0.5),
        "decision_tree": DecisionTreeRegressor(random_state=42),
        "random_forest": RandomForestRegressor(random_state=42),
        "knn": KNeighborsRegressor(n_neighbors=5),
        "svr": SVR(kernel="rbf"),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "mae": mean_absolute_error(y_test, pred),
            "rmse": np.sqrt(mean_squared_error(y_test, pred)),
            "r2": r2_score(y_test, pred),
        }
    return results


# -----------------------------------------------------------------------------
# 12. Classification algorithms
# -----------------------------------------------------------------------------

def classification_examples(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "logistic_regression": LogisticRegression(max_iter=500),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "gaussian_nb": GaussianNB(),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(random_state=42),
        "svm": SVC(probability=True, random_state=42),
        "adaboost": AdaBoostClassifier(random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob) if prob is not None else None,
        }
    return results


# -----------------------------------------------------------------------------
# 13. Unsupervised learning
# -----------------------------------------------------------------------------

def unsupervised_examples(X):
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels_kmeans = kmeans.fit_predict(X)

    agg = AgglomerativeClustering(n_clusters=3)
    labels_agg = agg.fit_predict(X)

    silhouette = silhouette_score(X, labels_kmeans)
    return {"kmeans": labels_kmeans, "agg": labels_agg, "silhouette": silhouette}


# -----------------------------------------------------------------------------
# 14. Ensemble learning and hyperparameter tuning
# -----------------------------------------------------------------------------

def ensemble_and_tuning(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    bagging_clf = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=20, random_state=42)
    bagging_clf.fit(X_train, y_train)
    bagging_pred = bagging_clf.predict(X_test)

    voting = VotingClassifier(estimators=[
        ("rf", RandomForestClassifier(random_state=42)),
        ("lr", LogisticRegression(max_iter=500)),
    ])
    voting.fit(X_train, y_train)
    voting_pred = voting.predict(X_test)

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5, None],
    }
    rf = RandomForestClassifier(random_state=42)
    search = GridSearchCV(rf, param_grid, cv=3, scoring="f1")
    search.fit(X_train, y_train)

    return {
        "bagging_accuracy": accuracy_score(y_test, bagging_pred),
        "voting_accuracy": accuracy_score(y_test, voting_pred),
        "best_params": search.best_params_,
    }


# -----------------------------------------------------------------------------
# 15. Model evaluation and diagnostics
# -----------------------------------------------------------------------------

def model_evaluation_examples(y_true, y_pred, y_score):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1:", f1_score(y_true, y_pred, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title("ROC curve")
    plt.show()

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    print("Average precision:", average_precision_score(y_true, y_score))


# -----------------------------------------------------------------------------
# 16. Imbalanced data handling
# -----------------------------------------------------------------------------

def imbalance_handling(X, y):
    if SMOTE is None:
        print("Install imbalanced-learn to run the SMOTE example.")
        return X, y

    sampler = SMOTE(random_state=42)
    X_bal, y_bal = sampler.fit_resample(X, y)
    print("Original samples:", len(X), "Balanced samples:", len(X_bal))
    return X_bal, y_bal


# -----------------------------------------------------------------------------
# 17. Deployment and saving models
# -----------------------------------------------------------------------------

def save_and_load_model(model, path="model.joblib"):
    dump(model, path)
    loaded = load(path)
    return loaded


# -----------------------------------------------------------------------------
# 18. End-to-end scikit-learn workflow
# -----------------------------------------------------------------------------

def end_to_end_pipeline(df):
    X = df.drop(columns=["target"])
    y = df["target"]

    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
    ])

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(random_state=42)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_pipeline.fit(X_train, y_train)
    pred = model_pipeline.predict(X_test)
    print(classification_report(y_test, pred))
    return model_pipeline


# -----------------------------------------------------------------------------
# 19. Example dataset creation
# -----------------------------------------------------------------------------

def create_example_dataset():
    np.random.seed(42)
    n = 400
    data = {
        "age": np.random.randint(18, 70, n),
        "salary": np.random.randint(20000, 150000, n),
        "city": np.random.choice(["new york", "london", "tokyo", "paris"], n),
        "gender": np.random.choice(["male", "female"], n),
        "size": np.random.choice(["small", "medium", "large"], n),
        "date": pd.date_range("2020-01-01", periods=n, freq="D") + pd.to_timedelta(np.random.randint(0, 365, n), unit="D"),
    }
    data["target"] = (data["salary"] > 80000).astype(int)
    return pd.DataFrame(data)


# -----------------------------------------------------------------------------
# 20. Run all examples
# -----------------------------------------------------------------------------

def main():
    df = create_example_dataset()
    quick_eda(df)

    df_clean = data_cleaning_examples(df)
    df_missing, _ = missing_value_examples(df_clean)
    df_cat = categorical_analysis(df_clean)

    X_enc, ohe_df, dummy_df = categorical_encoding_examples(df_cat)
    feature_scaling_examples(df_cat)
    df_fe, poly_features = feature_engineering_examples(df_cat)

    X = df_fe[["age", "salary", "city_frequency"]].copy()
    y = df_fe["target"].copy()
    X_sel, X_rfe, X_pca = feature_selection_examples(X, y)

    split_and_cross_validate(X, y)
    regression_results = regression_examples(X, y)
    classification_results = classification_examples(X, y)
    unsup = unsupervised_examples(X)
    ensemble_results = ensemble_and_tuning(X, y)

    X_bal, y_bal = imbalance_handling(X, y)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_bal, y_bal)
    save_and_load_model(model, "model.joblib")

    end_to_end_pipeline(df_clean)

    print("Regression results:", regression_results)
    print("Classification results:", classification_results)
    print("Unsupervised clustering:", unsup)
    print("Ensemble and tuning:", ensemble_results)


if __name__ == "__main__":
    main()
