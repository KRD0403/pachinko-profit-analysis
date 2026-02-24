import pandas as pd

def load_and_preprocess(path):

    df = pd.read_csv(path)

    # 列名整理
    df.columns = df.columns.str.strip()

    # 日付変換
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 金額カラムを数値化
    yen_cols = ["gross_profit_yen", "sales_yen", "net_profit_yen"]
    for col in yen_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .astype(float)
            )

    # 欠損削除
    df = df.dropna()

    #0除算防止
    df["sales_per_customer"] = df["sales_yen"] / df["customers"].replace(0, 1)
    df["sales_per_hour"] = df["sales_yen"] / df["operating_hours"].replace(0, 1)
    df["sales_per_spin"] = df["sales_yen"] / df["spins"].replace(0, 1)
     # 利益率
    df["profit_margin"] = df["net_profit_yen"] / df["sales_yen"].replace(0, 1)   
  
    target_col = "profit_margin"
    return df
