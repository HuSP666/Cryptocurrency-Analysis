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

def process_crypto_analysis(data_folder, window_days=3, mutation_threshold=0.10,
                            start_date=None, end_date=None):
    """
    加密货币数据分析主函数

    Parameters
    ----------
    data_folder : str
        存放CSV文件的文件夹路径
    window_days : int
        突变检测窗口天数（1～3天）
    mutation_threshold : float
        价格突变阈值（如0.10表示10%）
    start_date : str or None
        分析起始日期，格式 'YYYY-MM-DD'，不设限则为None
    end_date : str or None
        分析结束日期，格式 'YYYY-MM-DD'，不设限则为None
    """
    all_files = glob.glob(os.path.join(data_folder, "*.csv"))
    if not all_files:
        print("未找到 CSV 文件。")
        return

    summary_list = []
    mutation_list = []
    returns_dict = {}
    market_caps = {}
    corr_volume_ret = {}  # 存储每个币种的成交量-收益率相关系数

    print(f"正在分析 {len(all_files)} 个币种...")

    # 日期范围预处理
    if start_date:
        start_date = pd.to_datetime(start_date)
    if end_date:
        end_date = pd.to_datetime(end_date)

    for file in all_files:
        coin_name = os.path.basename(file).split('_')[0]
        # 读取数据 (分号分隔)
        df = pd.read_csv(file, sep=';', parse_dates=['timeOpen'])
        df = df.sort_values('timeOpen').reset_index(drop=True)

        # ---- 新增：日期范围筛选 ----
        if start_date:
            df = df[df['timeOpen'] >= start_date]
        if end_date:
            df = df[df['timeOpen'] <= end_date]
        if df.empty:
            print(f"警告：{coin_name} 在指定日期范围内无数据，已跳过。")
            continue

        # 基础统计
        start_price = df['open'].iloc[0]
        end_price = df['close'].iloc[-1]
        max_p, min_p = df['high'].max(), df['low'].min()
        avg_cap = df['marketCap'].mean()

        # 记录市值标签
        if avg_cap >= 1e9:
            cap_str = f"{avg_cap/1e9:.1f}B"
        else:
            cap_str = f"{avg_cap/1e6:.1f}M"
        market_caps[coin_name] = cap_str

        # ---- 新增：计算日收益率与交易量占比的相关性 ----
        # 日收益率（相对于前一日收盘价）
        df['daily_return'] = df['close'].pct_change()
        # 交易量占市值比例
        df['vol_ratio'] = df['volume'] / df['marketCap']

        # 剔除无效值（NaN或inf）
        valid_mask = df['daily_return'].notna() & df['vol_ratio'].notna() & np.isfinite(df['vol_ratio'])
        valid_df = df[valid_mask]
        if len(valid_df) >= 2:  # 至少两个有效交易日
            corr_val = valid_df['daily_return'].corr(valid_df['vol_ratio'], method='pearson')
        else:
            corr_val = np.nan
        corr_volume_ret[coin_name] = corr_val

        summary_list.append({
            '币种': coin_name,
            '市值': avg_cap,
            '最高价': max_p,
            '最低价': min_p,
            '期初价格': start_price,
            '期末价格': end_price,
            '总涨跌幅(%)': round(((end_price - start_price) / start_price) * 100, 2),
            '成交量与收益率相关性': round(corr_val, 4) if not np.isnan(corr_val) else '数据不足'
        })

        # --- 突变筛选逻辑 (1, 2, 3日跨度) ---
        for i in range(1, len(df)):
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
                        break

        # 记录滑动收益率用于相关性分析（仍使用3日窗口）
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
    if not returns_df.empty:
        corr_matrix = returns_df.corr(method='spearman')
        if len(corr_matrix) > 1:
            dist = pdist(corr_matrix)
            link = linkage(dist, method='ward')
            order = leaves_list(link)
            corr_matrix = corr_matrix.iloc[order, order]

        y_labels = [f"[{market_caps[col]}] {col}" for col in corr_matrix.index]
        x_labels = list(corr_matrix.columns)

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0,
                    xticklabels=x_labels, yticklabels=y_labels,
                    annot_kws={"size": 8}, cbar_kws={'label': '相关系数'})
        plt.title(f'加密货币相关性聚类热力图 (基于{window_days}日滚动窗口)', fontsize=15)
        plt.tight_layout()
        plt.show()

    # ---- 新增：控制台输出总体相关性结论 ----
    valid_corrs = [v for v in corr_volume_ret.values() if not np.isnan(v)]
    if valid_corrs:
        avg_corr = np.mean(valid_corrs)
        if avg_corr > 0.05:
            tendency = "正相关"
        elif avg_corr < -0.05:
            tendency = "负相关"
        else:
            tendency = "无明显线性相关"
        print(f"\n【总体成交量-收益率相关性】")
        print(f"有效币种数量: {len(valid_corrs)}")
        print(f"平均相关系数: {avg_corr:.4f}")
        print(f"结论: 整体呈 {tendency}。")
    else:
        print("\n【总体成交量-收益率相关性】无有效数据。")

    print("\n分析完成！")
    print("1. 已生成统计概览: crypto_summary_stats.xlsx")
    print("2. 已生成突变报告: crypto_mutations_report.xlsx")


# 执行示例（可自行修改日期范围）
if __name__ == "__main__":
    # 使用示例：指定起始和结束日期（不设限则传None）
    process_crypto_analysis(
        data_folder='E:\py',
        window_days=3,
        mutation_threshold=0.15,
        start_date=None,   # 修改为你要分析的起始日期
        end_date=None      # 修改为你要分析的结束日期
    )
