import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 设置图表支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False

def analyze_crypto_data(data_folder, window_days=3):
    """
    分析加密货币数据：最值、涨跌幅、以及基于滑动窗口的相关性
    :param data_folder: 存放 CSV 文件的文件夹路径
    :param window_days: 定义“几天内皆认为相关”的时间窗口（默认 3 天）
    """
    all_files = glob.glob(os.path.join(data_folder, "*.csv"))
    
    if not all_files:
        print("未找到任何 CSV 文件，请检查文件夹路径。")
        return

    summary_stats = []
    combined_returns = pd.DataFrame()

    print(f"正在处理 {len(all_files)} 个文件...")

    for file in all_files:
        # 从文件名提取币种名称，例如 "Dai_2026_1_1..." 提取出 "Dai"
        coin_name = os.path.basename(file).split('_')[0]
        
        # 读取数据，注意分隔符为分号
        df = pd.read_csv(file, sep=';', parse_dates=['timeOpen'])
        
        # 按时间正序排列（从过去到现在）
        df = df.sort_values('timeOpen').reset_index(drop=True)
        
        # 将时间列设为索引，并统一为日期格式（去除时区和时间）
        df['date'] = df['timeOpen'].dt.date
        df.set_index('date', inplace=True)
        
        # 1. 计算个体统计数据（最值与绝对涨跌幅）
        max_price = df['high'].max()
        min_price = df['low'].min()
        start_price = df['open'].iloc[0]
        end_price = df['close'].iloc[-1]
        total_amplitude_pct = ((end_price - start_price) / start_price) * 100
        
        summary_stats.append({
            '币种': coin_name,
            '最高价': max_price,
            '最低价': min_price,
            '期初价格': start_price,
            '期末价格': end_price,
            '区间总涨跌幅(%)': round(total_amplitude_pct, 2)
        })
        
        # 2. 计算用于相关性分析的滑动窗口收益率
        # pct_change(periods=window_days) 计算例如 3 天内的累计涨跌幅
        # 这完美解决了“几天内皆认为相关”的问题
        rolling_return = df['close'].pct_change(periods=window_days)
        
        # 将该币种的滑动收益率加入主数据框
        combined_returns[coin_name] = rolling_return

    # --- 输出个体分析结果 ---
    summary_df = pd.DataFrame(summary_stats)
    print("\n" + "="*40)
    print("加密货币个体统计概览:")
    print("="*40)
    print(summary_df.to_string(index=False))
    
    # --- 计算相关性矩阵 ---
    # 去除缺失值（由于计算了滑动窗口，前几天会是 NaN）
    combined_returns.dropna(inplace=True)
    
    # 使用 Spearman 相关系数，对异常值更鲁棒，适合加密货币的剧烈波动
    corr_matrix = combined_returns.corr(method='spearman')
    
    print("\n" + "="*40)
    print(f"跨币种相关性分析 (基于 {window_days} 天滑动窗口):")
    print("="*40)
    
    # 找出最高和最低相关的币种对
    # 将对角线（自己和自己相关度为1）设为 NaN 以便排除
    corr_matrix_no_diag = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # 转换为一维格式以便排序
    corr_pairs = corr_matrix_no_diag.stack().reset_index()
    corr_pairs.columns = ['币种_A', '币种_B', '相关系数']
    corr_pairs = corr_pairs.sort_values(by='相关系数', ascending=False)
    
    print("\n[高度正相关] (同涨同跌):")
    print(corr_pairs.head(5).to_string(index=False))
    
    print("\n[不相关或负相关] (走势独立或相反):")
    print(corr_pairs.tail(5).to_string(index=False))

    # --- 绘制相关性热力图 ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=.5)
    plt.title(f'加密货币 {window_days}天滚动收益率 相关性热力图')
    plt.tight_layout()
    plt.show()

# --- 运行代码 ---
# 请将 'E:\py' 替换为你实际存放这 20+ 个 CSV 文件的文件夹路径
if __name__ == "__main__":
    # 假设 CSV 文件都在当前目录下的 data 文件夹中
    analyze_crypto_data('E:\py', window_days=3)
