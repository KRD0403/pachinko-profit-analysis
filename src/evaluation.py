import matplotlib.pyplot as plt
import seaborn as sns

def plot_net_profit_distribution(df):
    sns.histplot(df["net_profit_yen"], bins=30, kde=True)
    plt.title("net_Profit Distribution")
    plt.show()

def plot_correlation(df):
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluate_regression(model, X_test, y_test):

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"R2: {r2:.3f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    # 実測 vs 予測
    plt.figure()
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()]
    )
    plt.xlabel("Actual Net Profit")
    plt.ylabel("Predicted Net Profit")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names):
    importances = model.named_steps["model"].feature_importances_
    plt.figure()
    sns.barplot(x=importances, y=feature_names)
    plt.title("Feature Importance")
    plt.show()