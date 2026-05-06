import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def _parse_date(date_value):
    if date_value is None or date_value == "":
        return None
    return pd.to_datetime(date_value).tz_localize(None) if pd.Timestamp(date_value).tzinfo else pd.to_datetime(date_value)


def _load_and_filter_csv(file_path, start_date=None, end_date=None):
    df = pd.read_csv(file_path, sep=';')
    if 'timeOpen' not in df.columns:
        raise ValueError(f"{os.path.basename(file_path)} 缺少 timeOpen 列")
    df['timeOpen'] = pd.to_datetime(df['timeOpen'], errors='coerce', utc=True).dt.tz_convert(None)
    df = df.dropna(subset=['timeOpen']).sort_values('timeOpen').reset_index(drop=True)
    if start_date is not None:
        df = df[df['timeOpen'] >= pd.to_datetime(start_date)]
    if end_date is not None:
        end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        df = df[df['timeOpen'] <= end_ts]
    return df.reset_index(drop=True)


def process_crypto_analysis(
    data_folder,
    window_days=3,
    mutation_threshold=0.10,
    analysis_start_date=None,
    analysis_end_date=None
):
    all_files = glob.glob(os.path.join(data_folder, "*.csv"))
    if not all_files:
        print("未找到 CSV 文件。")
        return

    summary_list = []
    mutation_list = []
    returns_dict = {}
    market_caps = []
    daily_corr_list = []
    combined_daily_rows = []
    coin_specific_corrs = []
    long_term_records = []

    print(f"正在分析 {len(all_files)} 个币种...")
    if analysis_start_date or analysis_end_date:
        print(f"分析日期范围：{analysis_start_date or '最早'} ~ {analysis_end_date or '最晚'}")

    for file in all_files:
        coin_name = os.path.basename(file).split('_')[0]

        try:
            df = _load_and_filter_csv(file, analysis_start_date, analysis_end_date)
        except Exception as e:
            print(f"跳过 {coin_name}：{e}")
            continue

        if df.empty:
            print(f"跳过 {coin_name}：筛选后无数据。")
            continue

        # 基础统计
        start_price = df['open'].iloc[0]
        end_price = df['close'].iloc[-1]
        max_p, min_p = df['high'].max(), df['low'].min()
        avg_cap = df['marketCap'].mean()
        avg_volume = df['volume'].mean()

        if avg_cap >= 1e9:
            cap_str = f"{avg_cap/1e9:.1f}B"
        else:
            cap_str = f"{avg_cap/1e6:.1f}M"
        market_caps.append((coin_name, cap_str))

        summary_list.append({
            '币种': coin_name,
            '数据起始日': df['timeOpen'].iloc[0].strftime('%Y-%m-%d'),
            '数据结束日': df['timeOpen'].iloc[-1].strftime('%Y-%m-%d'),
            '样本数': len(df),
            '市值': avg_cap,
            '平均成交量': avg_volume,
            '最高价': max_p,
            '最低价': min_p,
            '期初价格': start_price,
            '期末价格': end_price,
            '总涨跌幅(%)': round(((end_price - start_price) / start_price) * 100, 2)
        })

        # 突变检测
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

        # 日度数据
        df = df.copy()
        df['date'] = df['timeOpen'].dt.date
        df['daily_return'] = df['close'].pct_change()
        df['volume_to_marketcap'] = np.where(df['marketCap'] > 0, df['volume'] / df['marketCap'], np.nan)

        # 日度相关性（换手率）
        corr_df = df[['date', 'daily_return', 'volume_to_marketcap']].dropna()
        if not corr_df.empty:
            corr_coef, p_value = spearmanr(corr_df['daily_return'], corr_df['volume_to_marketcap'])
            daily_corr_list.append({
                '币种': coin_name,
                '样本数': int(len(corr_df)),
                'Spearman相关系数': corr_coef,
                'p值': p_value,
                '结论': '相关' if p_value < 0.05 else '不显著相关'
            })
            combined_daily_rows.append(corr_df.assign(币种=coin_name))

        # 单币种：日收益 vs 市值、vs 换手率
        df['turnover'] = df['volume_to_marketcap']
        valid = df[['daily_return', 'marketCap', 'turnover']].dropna()
        if len(valid) >= 5:
            r_cap, p_cap = spearmanr(valid['daily_return'], valid['marketCap'])
            r_turn, p_turn = spearmanr(valid['daily_return'], valid['turnover'])
            coin_specific_corrs.append({
                '币种': coin_name,
                '样本数': len(valid),
                '日收益vs市值_Spearman_r': r_cap,
                '日收益vs市值_p值': p_cap,
                '市值相关性结论': '相关' if p_cap < 0.05 else '不显著相关',
                '日收益vs换手率_Spearman_r': r_turn,
                '日收益vs换手率_p值': p_turn,
                '换手率相关性结论': '相关' if p_turn < 0.05 else '不显著相关'
            })
        else:
            coin_specific_corrs.append({
                '币种': coin_name,
                '样本数': len(valid),
                '日收益vs市值_Spearman_r': np.nan,
                '日收益vs市值_p值': np.nan,
                '市值相关性结论': '数据不足',
                '日收益vs换手率_Spearman_r': np.nan,
                '日收益vs换手率_p值': np.nan,
                '换手率相关性结论': '数据不足'
            })

        # 滚动窗口收益率
        df = df.set_index('date')
        returns_dict[coin_name] = df['close'].pct_change(periods=window_days)

        # 长期指标（均值、波动率、平均市值、平均成交量、平均换手率）
        daily_returns = df['daily_return'].dropna()
        mean_ret = daily_returns.mean() if len(daily_returns) >= 2 else np.nan
        var_ret = daily_returns.var() if len(daily_returns) >= 2 else np.nan
        avg_turnover = df['volume_to_marketcap'].mean()  # 平均换手率

        long_term_records.append({
            '币种': coin_name,
            '平均日收益率': mean_ret,
            '日收益方差': var_ret,
            '波动率(标准差)': np.sqrt(var_ret) if pd.notna(var_ret) else np.nan,
            '平均市值': avg_cap,
            '平均交易量': avg_volume,
            '平均换手率': avg_turnover
        })

    if not summary_list:
        print("筛选后没有可分析的数据。")
        return

    # ===== 原有输出 =====
    summary_df = pd.DataFrame(summary_list).sort_values('币种')
    daily_corr_df = pd.DataFrame(daily_corr_list).sort_values('币种') if daily_corr_list else pd.DataFrame()

    # 总体结论
    overall_corr_coef = np.nan
    overall_corr_p = np.nan
    if combined_daily_rows:
        all_daily = pd.concat(combined_daily_rows, ignore_index=True)
        overall_corr_coef, overall_corr_p = spearmanr(all_daily['daily_return'], all_daily['volume_to_marketcap'])

    with pd.ExcelWriter("crypto_summary_stats.xlsx", engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="统计概览", index=False)
        if not daily_corr_df.empty:
            daily_corr_df.to_excel(writer, sheet_name="日度相关性", index=False)
        else:
            pd.DataFrame().to_excel(writer, sheet_name="日度相关性", index=False)
        pd.DataFrame([{
            '总体Spearman相关系数': overall_corr_coef,
            '总体p值': overall_corr_p,
            '总体结论': '相关' if (pd.notna(overall_corr_p) and overall_corr_p < 0.05) else '不显著相关'
        }]).to_excel(writer, sheet_name="总体结论", index=False)

    # 突变报告
    mutation_df = pd.DataFrame(mutation_list)
    if not mutation_df.empty:
        mutation_df = mutation_df.sort_values(by='结束时间')
    with pd.ExcelWriter("crypto_mutations_report.xlsx", engine="openpyxl") as writer:
        mutation_df.to_excel(writer, sheet_name="突变报告", index=False)

    # 相关性热力图
    returns_df = pd.DataFrame(returns_dict).dropna()
    corr_matrix = returns_df.corr(method='spearman')
    if len(corr_matrix) > 1:
        dist = pdist(corr_matrix)
        link = linkage(dist, method='ward')
        order = leaves_list(link)
        corr_matrix = corr_matrix.iloc[order, order]
    y_labels = []
    for col in corr_matrix.columns:
        cap_match = next((cap for coin, cap in market_caps if coin == col), "")
        y_labels.append(f"[{cap_match}] {col}")
    x_labels = list(corr_matrix.columns)

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        corr_matrix, annot=True, cmap='RdYlGn', center=0,
        xticklabels=x_labels, yticklabels=y_labels,
        annot_kws={"size": 8}, cbar_kws={'label': '相关系数'}
    )
    for label in ax.get_yticklabels():
        label.set_fontsize(10)
    plt.title(f'加密货币相关性聚类热力图 (基于{window_days}日滚动窗口)', fontsize=15)
    plt.tight_layout()

   # ===== 新增分析：一、均值-方差散点图 =====
    long_df = pd.DataFrame(long_term_records).dropna(subset=['平均日收益率', '波动率(标准差)', '平均市值', '平均换手率'])
    if not long_df.empty:
        plt.figure(figsize=(10, 8))
        # 气泡大小放大：最大气泡对应最大市值，缩放至800点
        size = long_df['平均市值'] / long_df['平均市值'].max() * 800
        scatter = plt.scatter(
            long_df['平均日收益率'], long_df['波动率(标准差)'],
            s=size, c=long_df['平均换手率'], cmap='viridis', alpha=0.7,
            edgecolors='k', linewidth=0.5
        )
        plt.colorbar(scatter, label='平均换手率')

        # 使用 adjustText 防止文字重叠
        try:
            from adjustText import adjust_text
            texts = []
            for idx, row in long_df.iterrows():
                texts.append(
                    plt.text(row['平均日收益率'], row['波动率(标准差)'], row['币种'],
                             fontsize=9, ha='center', va='center')
                )
            adjust_text(texts, force_text=0.5, expand_text=(1.2, 1.5),
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
            print("已启用文字自动排布（adjustText）")
        except ImportError:
            print("提示：未安装 adjustText 库，文字可能重叠。请执行 pip install adjustText 后重试。")
            # 回退到简单标注（可能重叠）
            for idx, row in long_df.iterrows():
                plt.annotate(row['币种'], (row['平均日收益率'], row['波动率(标准差)']),
                             fontsize=9, alpha=0.8, textcoords="offset points", xytext=(5, 5))

        plt.xlabel('平均日收益率')
        plt.ylabel('日收益波动率（标准差）')
        plt.title('加密货币个体特征：均值-方差散点图（气泡大小=平均市值，颜色=平均换手率）')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("crypto_mean_variance_scatter.png", dpi=150)
        plt.show()
    else:
        print("警告：没有足够数据绘制均值-方差散点图。")

    # ===== 新增分析：二、波动率/均值与市值、换手率的相关性矩阵 =====
    # 构建相关性矩阵（Spearman）
    metric_cols = ['平均日收益率', '波动率(标准差)', '平均市值', '平均换手率']
    available = [col for col in metric_cols if col in long_df.columns and long_df[col].notna().sum() >= 3]
    cross_corr_additional = None
    if len(available) >= 2:
        cross_corr_additional = long_df[available].corr(method='spearman')

    # 将原来的长期指标相关矩阵也保留（包括交易量）
    metric_cols_full = ['平均日收益率', '波动率(标准差)', '平均市值', '平均交易量', '平均换手率']
    available_full = [col for col in metric_cols_full if col in long_df.columns and long_df[col].notna().sum() >= 3]
    cross_corr_full = None
    if len(available_full) >= 2:
        cross_corr_full = long_df[available_full].corr(method='spearman')

    # 保存到高级相关性文件
    with pd.ExcelWriter("crypto_advanced_correlations.xlsx", engine="openpyxl") as writer:
        # 原有长期指标明细
        long_df.to_excel(writer, sheet_name="长期指标明细", index=False)
        # 新增：波动率/均值与市值/换手率相关矩阵
        if cross_corr_additional is not None:
            cross_corr_additional.to_excel(writer, sheet_name="波动率均值vs市值换手率相关")
        # 更全面的相关矩阵（含交易量）
        if cross_corr_full is not None:
            cross_corr_full.to_excel(writer, sheet_name="长期指标完整相关矩阵")

    # 单币种相关性输出
    coin_specific_df = pd.DataFrame(coin_specific_corrs).sort_values('币种')
    with pd.ExcelWriter("crypto_coin_specific_correlations.xlsx", engine="openpyxl") as writer:
        coin_specific_df.to_excel(writer, sheet_name="单币种相关性", index=False)

    print("\n分析完成！")
    print("1. 已生成统计概览: crypto_summary_stats.xlsx")
    print("2. 已生成突变报告: crypto_mutations_report.xlsx")
    print("3. 已生成高级相关性分析: crypto_advanced_correlations.xlsx")
    print("4. 已生成单币种相关性分析: crypto_coin_specific_correlations.xlsx")
    print("5. 已生成均值-方差散点图: crypto_mean_variance_scatter.png")
    if pd.notna(overall_corr_coef):
        print(f"总体相关系数（涨跌幅 vs 交易量/市值）：{overall_corr_coef:.4f}，p={overall_corr_p:.4g}")
    else:
        print("总体结论：无足够数据计算相关性。")

    plt.show()


if __name__ == "__main__":
    process_crypto_analysis(
        'E:\\py',
        window_days=3,
        mutation_threshold=0.15,
        analysis_start_date=None,
        analysis_end_date=None
    )
