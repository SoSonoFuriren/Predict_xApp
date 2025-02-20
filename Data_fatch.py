import numpy as np
import pandas as pd

def generate_time_series_multiFeature(n_samples, n_steps):
    np.random.seed(42)
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, n_samples, 1)
    time = np.linspace(0, 1, n_steps)
    
    # 原始的時間序列 (正弦波)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))
    series += 0.5 * np.sin((time - offsets2) * (freq2 * 20 + 20))
    series += 0.1 * (np.random.rand(n_samples, n_steps) - 0.5)  # 加入少許噪音
    
    # 額外特徵 1: 時間索引
    time_index = np.tile(time, (n_samples, 1))  
    
    # 額外特徵 2: 過去 3 個時間步的移動平均
    moving_avg = np.zeros_like(series)
    for i in range(3, n_steps):  
        moving_avg[:, i] = np.mean(series[:, i-3:i], axis=1)  # 計算移動平均
    
    # 合併三個特徵
    features = np.stack([series, time_index, moving_avg], axis=-1)  # shape: (n_samples, n_steps, 3)
    return features

# 產生簡單的時間序列數據（正弦波）
def generate_time_series_singleFeature(n_samples, n_steps):
    np.random.seed(42)
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, n_samples, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))
    series += 0.5 * np.sin((time - offsets2) * (freq2 * 20 + 20))
    series += 0.1 * (np.random.rand(n_samples, n_steps) - 0.5)  # 加入少許噪音
    return series[..., np.newaxis]

def Read_VIAVI_Cell_Metrics(fileName, cellName = "S1/N78/C1", metrics = ["DRB.UEThpDl", "RRU.PrbUsedDl"]):
    # 讀取 CSV 檔案
    df = pd.read_csv(fileName)  # 假設是 TAB 分隔
    # print(df["Viavi.Cell.Name"].unique())

    # 篩選出 "S1/N78/C1" 的資料，並且只保留指定的欄位
    filtered_df = df[df["Viavi.Cell.Name"] == cellName][metrics]

    return filtered_df