import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import Data_fatch as df

# 產生簡單的時間序列數據（正弦波） + 額外特徵

# 生成訓練數據
n_samples = 1000
n_steps = 50
series = df.generate_time_series_multiFeature(n_samples, n_steps + 1)
print(series.shape)

# 切分資料集
X_train, y_train = series[:700, :-1], series[:700, -1, 0]  # 最後一個值作為目標值 (只取原始序列)
X_valid, y_valid = series[700:900, :-1], series[700:900, -1, 0]
X_test, y_test = series[900:, :-1], series[900:, -1, 0]

# 顯示前 5 筆訓練數據
print("前 5 筆訓練數據 (每筆包含 50 個時間步，每個時間步有 3 個特徵)：")
for i in range(2):
    print(f"\n第 {i+1} 筆 X_train (形狀: {X_train[i].shape})：")
    print(X_train[i])  # 輸入數據
    print(f"對應的 y_train：{y_train[i]}")  # 目標數據

# 建立 LSTM 模型（支援多特徵）
model = keras.Sequential([
    keras.layers.LSTM(50, return_sequences=False, input_shape=[n_steps, 3]),  
    keras.layers.Dense(1)  
])

# 編譯模型
model.compile(loss="mse", optimizer="adam")

# 訓練模型
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))

# 進行滾動預測（未來 50 個時間步）
n_future = 50
X_input = X_test[0].copy()
predictions = []

for _ in range(n_future):
    y_pred = model.predict(X_input[np.newaxis, :, :])[0, 0]
    predictions.append(y_pred)

    # 更新輸入數據
    X_input = np.roll(X_input, -1, axis=0)  
    X_input[-1, 0] = y_pred  # 更新原始序列
    X_input[-1, 1] = X_input[-2, 1] + (X_input[1, 1] - X_input[0, 1])  # 更新時間索引
    X_input[-1, 2] = np.mean(X_input[-3:, 0])  # 更新移動平均

# 繪製結果
plt.figure(figsize=(10, 5))
plt.plot(range(n_steps), X_test[0, :, 0], label="Input Sequence")
plt.plot(range(n_steps, n_steps + n_future), predictions, label="Predicted Future")
plt.legend()
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.title("LSTM Future Prediction with Extra Features")
plt.show()