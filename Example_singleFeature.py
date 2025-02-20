import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import Data_fatch as df

# 生成訓練數據
n_samples = 1000
n_steps = 50
series = df.generate_time_series_singleFeature(n_samples, n_steps + 1)
X_train, y_train = series[:700, :-1], series[:700, -1] # 取最後一比以前的值做訓練 # 最後一個值作為目標值
X_valid, y_valid = series[700:900, :-1], series[700:900, -1]
X_test, y_test = series[900:, :-1], series[900:, -1]

actvals = []
actval = series[900, :]
# print(actval.shape)
for i in range(0, n_steps+1):
    actvals.append(actval[i])
print(len(actvals))


# # 可視化部分訓練數據
# plt.figure(figsize=(10, 5))
# for i in range(1):
#     plt.plot(X_test[i], label=f"Sample {i+1}")
#     print(f"Sample {i+1}:", X_train[i].flatten())
# # print("Shape of X_train:", y_train.shape)
# # print("Shape of y_train:", y_train.flatten())
# plt.xlabel("Time Steps")
# plt.ylabel("Value")
# plt.title("Generated Time Series Data")
# plt.legend()
# plt.show()

# 建立 LSTM 模型
model = keras.Sequential([
    keras.layers.LSTM(50, return_sequences=False, input_shape=[n_steps, 1]),
    keras.layers.Dense(1)
])

# 編譯模型
model.compile(loss="mse", optimizer="adam")

# 訓練模型
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))


n_future = 50  # 想要預測的未來時間步數
X_input = X_test[0].copy()  # 取得第一個測試樣本
predictions = []

for _ in range(n_future):
    y_pred = model.predict(X_input[np.newaxis, :, :])[0, 0]  # 預測下一個點
    predictions.append(y_pred)  # 存儲預測值
    X_input = np.roll(X_input, -1)  # 滑動視窗
    X_input[-1] = y_pred  # 用預測值填入最後一個時間步


# 繪製結果
plt.figure(figsize=(10, 5))
plt.plot(range(n_steps + 1), actvals, label="Input Sequence")  # 輸入的 X_test[0]
plt.plot(range(n_steps, n_steps + n_future), predictions, label="Predicted Future")
plt.legend()
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.title("LSTM Future Prediction")
plt.show()


# # 預測
# y_pred = model.predict(X_test[0:1:, :-1])

# print("True Values:", y_test[0])
# print("Predicted Values:", y_pred)
# # 繪製預測結果
# plt.figure(figsize=(10, 5))
# plt.plot(X_test[0], label="True Values")
# plt.plot(y_pred, label="Predictions")
# plt.legend()
# plt.xlabel("Time Steps")
# plt.ylabel("Value")
# plt.title("LSTM Time Series Prediction")
# plt.show()
