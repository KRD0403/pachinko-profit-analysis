📊 Pachinko Profit Analysis
Machine learning project to predict daily net profit in pachinko stores and analyze key profit drivers.
1. 背景・目的

本プロジェクトは、パチンコ店舗における利益構造の把握と主要因の特定を目的として実施した。
単日の営業データを用い、利益に影響を与える要因を定量的に分析し、改善施策の示唆を導出することを目標とした。

2. データ概要

・期間：8日
・台数：31台
・単位：台 × 日のレコード

主な変数：

変数名	          内容
operating_hours  営業時間
spins	　　　　  総回転数
customers	     来店客数
sales_yen	     売上
net_profit_yen   純利益（目的変数）
event_flag	     イベント有無
machine_type	 機種
weather	         天候
weekday	         曜日

3. 分析手法

前処理
・ColumnTransformer による統合前処理
・数値変数：StandardScaler
・カテゴリ変数：OneHotEncoder(handle_unknown="ignore")

モデル比較
以下の回帰モデルを比較した。
・Linear Regression
・Ridge Regression
・Lasso Regression
・RandomForestRegressor

評価方法
・Train/Test split（80/20）
・5-fold Cross Validation
・評価指標：R2 / RMSE / MAE

4. モデル結果

最良モデルは RandomForest であった。

指標	値
CV R2	0.49
Test R2	0.36
RMSE	33,559
MAE	　　20,282

線形モデルと比較して説明力が大きく向上しており、利益構造に非線形関係が含まれている可能性が示唆された。

5. 特徴量重要度分析

RandomForestの特徴量重要度より、利益に強い影響を与えている要因は以下であった。

特徴量	            重要度
sales_per_customer	0.55
sales_per_hour	    0.19
sales_per_spin	    0.16

分析結果の解釈
・来店客数よりも客単価が利益を強く支配
・単日視点では、集客施策よりも単価改善施策の方が利益向上に寄与する可能性
・イベント実施の直接的な影響は限定的

6. 考察

本分析は単日データに基づくものであり、短期的な利益構造を示している。

一方で、客単価を過度に高める施策は長期的には顧客離脱を招く可能性がある。
そのため、
[短期利益最大化と顧客維持のバランス設計]
が今後の重要な検討課題と考えられる。

7. 今後の課題
・週次・月次単位での再集計分析
・来店頻度・リピート率を考慮した長期モデル構築
・ハイパーパラメータ最適化の体系化
・可視化レポートの整備

8. 使用技術
・Python
・pandas
・scikit-learn
・Pipeline / ColumnTransformer
・RandomForestRegressor

9. 本プロジェクトで得られた知見・スキル
・前処理設計を含む機械学習パイプライン構築
・複数モデル比較およびCVによる汎化性能評価
・特徴量重要度を用いた構造解釈
・ビジネス視点を踏まえた示唆抽出
・モデルの限界を踏まえた課題整理

10. 実行方法

pip install -r requirements.txt
python main.py

※ 実行するとモデル学習および評価が行われ、結果はコンソールに出力されます。
※ 分析過程のEDAは notebooks フォルダに保存しています。
※ モデル評価時の可視化結果（予測値比較・残差分布は output フォルダに保存しています。

📊 Pachinko Profit Analysis

機械学習を用いてパチンコ店舗の日次純利益を予測し、利益構造の定量分析と改善示唆を行ったプロジェクト。