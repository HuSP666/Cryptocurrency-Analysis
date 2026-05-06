import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist

# 设置支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

#日期窗与突变率默认设置，设置在末尾
def process_crypto_analysis(data_folder, window_days=3, mutation_threshold=0.10):
    all_files = glob.glob(os.path.join(data_folder, "*.csv"))
    if not all_files:
        print("未找到 CSV 文件。")
        return

    summary_list = []
    mutation_list = []
    returns_dict = {}
    market_caps = {}

    print(f"正在分析 {len(all_files)} 个币种...")

    for file in all_files:
        coin_name = os.path.basename(file).split('_')[0]
        # 读取数据 (分号分隔)
        df = pd.read_csv(file, sep=';', parse_dates=['timeOpen'])
        df = df.sort_values('timeOpen').reset_index(drop=True)
        
        # 基础统计
        start_price = df['open'].iloc[0]
        end_price = df['close'].iloc[-1]
        max_p, min_p = df['high'].max(), df['low'].min()
        avg_cap = df['marketCap'].mean()
        
        # 记录市值（用于热力图标签）
        # 格式化为 B(十亿) 或 M(百万)
        if avg_cap >= 1e9:
            cap_str = f"{avg_cap/1e9:.1f}B"
        else:
            cap_str = f"{avg_cap/1e6:.1f}M"
        market_caps[coin_name] = cap_str

        summary_list.append({
            '币种': coin_name,
            '市值': avg_cap,
            '最高价': max_p,
            '最低价': min_p,
            '期初价格': start_price,
            '期末价格': end_price,
            '总涨跌幅(%)': round(((end_price - start_price) / start_price) * 100, 2)
        })

        # --- 突变筛选逻辑 (1, 2, 3日跨度) ---
        for i in range(1, len(df)):
            # 检查 1-3 天内的变化
            for lookback in range(1, window_days + 1):
                if i - lookback >= 0:
                    prev_idx = i - lookback
                    p_start = df.iloc[prev_idx]['open']
                    p_end = df.iloc[i]['close']
                    change = (p_end - p_start) / p_start
                    
                    if abs(change) >= mutation_threshold:
                        mutation_list.append({
                            '结束时间': df.iloc[i]['timeOpen'].strftime('%Y-%m-%d'),
                            '起始时间': df.iloc[prev_idx]['timeOpen'].strftime('%Y-%m-%d'),
                            '币种': coin_name,
                            '涨跌幅(%)': round(change * 100, 2),
                            '当前价格': df.iloc[i]['close'],
                            '当期市值': df.iloc[i]['marketCap']
                        })
                        # 找到该点最大的突变后即跳过当前lookback，避免重复记录同一天的微小差异
                        break 

        # 记录 3 日滑动收益率用于相关性分析
        df['date'] = df['timeOpen'].dt.date
        df.set_index('date', inplace=True)
        returns_dict[coin_name] = df['close'].pct_change(periods=window_days)

    # 1. 保存统计概览 Excel
    summary_df = pd.DataFrame(summary_list)
    summary_df.to_excel("crypto_summary_stats.xlsx", index=False)
    
    # 2. 保存突变报告 Excel
    mutation_df = pd.DataFrame(mutation_list)
    if not mutation_df.empty:
        mutation_df = mutation_df.sort_values(by='结束时间')
        mutation_df.to_excel("crypto_mutations_report.xlsx", index=False)

    # 3. 相关性聚类与绘图
    returns_df = pd.DataFrame(returns_dict).dropna()
    corr_matrix = returns_df.corr(method='spearman')

    # 执行层次聚类以重新排序
    if len(corr_matrix) > 1:
        # 计算距离并进行链接
        dist = pdist(corr_matrix)
        link = linkage(dist, method='ward')
        order = leaves_list(link)
        # 按聚类结果重排矩阵
        corr_matrix = corr_matrix.iloc[order, order]

    # 准备标签
    # Y轴：市值 (较小字体) + 币种
    y_labels = [f"[{market_caps[col]}] {col}" for col in corr_matrix.index]
    x_labels = list(corr_matrix.columns)

    # 绘图
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0,
                     xticklabels=x_labels, yticklabels=y_labels,
                     annot_kws={"size": 8}, cbar_kws={'label': '相关系数'})

    # 优化坐标轴（不可用）
    # plt.xticks(rotation=45, ha='left') # 横轴向上倾斜45度
    
    # 修改左侧市值标签的样式（通过设置特定标签的属性模拟）
    for label in ax.get_yticklabels():
        label.set_fontsize(10)
        # 可以在这里进一步对字符串做正则处理实现不同颜色，但matplotlib原生支持有限
        # 这里采用统一格式化 [Cap] Name

    plt.title(f'加密货币相关性聚类热力图 (基于{window_days}日滚动窗口)', fontsize=15)
    plt.tight_layout()
    
    print("\n分析完成！")
    print("1. 已生成统计概览: crypto_summary_stats.xlsx")
    print("2. 已生成突变报告: crypto_mutations_report.xlsx")
    plt.show()

# 执行
if __name__ == "__main__":
    # 确保 data 文件夹存在并放入了您的 CSV
    process_crypto_analysis('E:\py', window_days=3, mutation_threshold=0.15)
