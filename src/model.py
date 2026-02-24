from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def train_model(df, feature_cols, target_col, model_type="rf"):

    X = df[feature_cols]
    y = df[target_col]

    categorical_cols = ["machine_type", "weather", "weekday"]
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
        ]
    )

    if model_type == "linear":
        model = LinearRegression()

    elif model_type == "ridge":
        model = Ridge(alpha=1.0)

    elif model_type == "lasso":
        model = Lasso(alpha=0.1, max_iter=20000)

    elif model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )

    else:
        raise ValueError("Invalid model_type")

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test