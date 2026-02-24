from src.preprocessing import load_and_preprocess
from src.model import train_model
from src.evaluation import evaluate_regression
from src.config import DATA_PATH
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

def main():

    df = load_and_preprocess(DATA_PATH)

    feature_cols = [
       "operating_hours",
       "spins",
       "customers",
       "event_flag",
       "sales_per_customer",
       "sales_per_hour",
       "sales_per_spin",
       "machine_type",
       "weather",
       "weekday"
    ]

    target_col = "net_profit_yen"

    for model_type in ["linear", "ridge", "lasso", "rf"]:

        print(f"\n=== {model_type.upper()} ===")

        model, X_test, y_test = train_model(
            df, feature_cols, target_col, model_type=model_type
        )

        cv_scores = cross_val_score(
            model,
            df[feature_cols],
            df[target_col],
            cv=5,
            scoring="r2"
        )

        print("CV R2:", np.mean(cv_scores))

        evaluate_regression(model, X_test, y_test)

        if model_type == "rf":
            model_rf = model.named_steps["model"]
            feature_names = model.named_steps["preprocess"].get_feature_names_out()
            importances = model_rf.feature_importances_

            importance_df = (
                pd.DataFrame({
                    "feature": feature_names,
                    "importance": importances
                })
               .sort_values("importance", ascending=False)
        )

            print("\n=== Feature Importance (Top 10) ===")
            print(importance_df.head(10))
print("\nModel comparison completed.")

if __name__ == "__main__":
    main()