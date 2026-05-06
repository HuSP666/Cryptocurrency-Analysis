
import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

# 设置支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def _parse_date(date_value):
    """将日期参数统一转换为 pandas Timestamp（不带时区），支持 None / str / datetime。"""
    if date_value is None or date_value == "":
        return None
    return pd.to_datetime(date_value).tz_localize(None) if pd.Timestamp(date_value).tzinfo else pd.to_datetime(date_value)


def _load_and_filter_csv(file_path, start_date=None, end_date=None):
    """读取单个币种 CSV，并按日期范围过滤。"""
    df = pd.read_csv(file_path, sep=';')

    if 'timeOpen' not in df.columns:
        raise ValueError(f"{os.path.basename(file_path)} 缺少 timeOpen 列，无法解析日期。")

    # 兼容 ISO 时间格式，例如 2026-04-20T00:00:00.000Z
    df['timeOpen'] = pd.to_datetime(df['timeOpen'], errors='coerce', utc=True).dt.tz_convert(None)
    df = df.dropna(subset=['timeOpen']).sort_values('timeOpen').reset_index(drop=True)

    if start_date is not None:
        start_ts = pd.to_datetime(start_date)
        df = df[df['timeOpen'] >= start_ts]
    if end_date is not None:
        end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        df = df[df['timeOpen'] <= end_ts]

    return df.reset_index(drop=True)


# 日期窗与突变率默认设置，新增分析日期范围参数
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

        # 市值标签
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

        # 日度相关性：涨跌幅 vs 交易量/市值
        df = df.copy()
        df['date'] = df['timeOpen'].dt.date
        df['daily_return'] = df['close'].pct_change()
        df['volume_to_marketcap'] = np.where(
            df['marketCap'] > 0,
            df['volume'] / df['marketCap'],
            np.nan
        )

        corr_df = df[['date', 'daily_return', 'volume_to_marketcap']].dropna()
        if not corr_df.empty:
            corr_coef, p_value = spearmanr(corr_df['daily_return'], corr_df['volume_to_marketcap'])
            daily_corr_list.append({
                '币种': coin_name,
                '样本数': int(len(corr_df)),
                'Spearman相关系数': corr_coef,
                'p值': p_value,
                '结论': '相关' if (pd.notna(p_value) and p_value < 0.05) else '不显著相关'
            })

            combined_daily_rows.append(
                corr_df.assign(币种=coin_name)
            )

        # 用于币种之间的收益率相关性热力图
        df = df.set_index('date')
        returns_dict[coin_name] = df['close'].pct_change(periods=window_days)

    if not summary_list:
        print("筛选后没有可分析的数据。")
        return

    # 1. 保存统计概览 Excel（含相关性结果）
    summary_df = pd.DataFrame(summary_list).sort_values('币种')
    daily_corr_df = pd.DataFrame(daily_corr_list).sort_values('币种') if daily_corr_list else pd.DataFrame()

    mutation_df = pd.DataFrame(mutation_list)
    if not mutation_df.empty:
        mutation_df = mutation_df.sort_values(by='结束时间')

    overall_corr_text = "无足够数据"
    overall_corr_coef = np.nan
    overall_corr_p = np.nan

    if combined_daily_rows:
        all_daily_df = pd.concat(combined_daily_rows, ignore_index=True)
        overall_corr_coef, overall_corr_p = spearmanr(all_daily_df['daily_return'], all_daily_df['volume_to_marketcap'])
        overall_corr_text = "相关" if (pd.notna(overall_corr_p) and overall_corr_p < 0.05) else "不显著相关"

    overview_file = "crypto_summary_stats.xlsx"
    with pd.ExcelWriter(overview_file, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="统计概览", index=False)
        if not daily_corr_df.empty:
            daily_corr_df.to_excel(writer, sheet_name="日度相关性", index=False)
        else:
            pd.DataFrame([{
                '币种': '',
                '样本数': 0,
                'Spearman相关系数': np.nan,
                'p值': np.nan,
                '结论': '无数据'
            }]).to_excel(writer, sheet_name="日度相关性", index=False)

        # 额外在概览页下方补充总体结论
        meta_df = pd.DataFrame([{
            '总体Spearman相关系数': overall_corr_coef,
            '总体p值': overall_corr_p,
            '总体结论': overall_corr_text
        }])
        meta_df.to_excel(writer, sheet_name="总体结论", index=False)

    # 2. 保存突变报告 Excel
    mutation_file = "crypto_mutations_report.xlsx"
    with pd.ExcelWriter(mutation_file, engine="openpyxl") as writer:
        if not mutation_df.empty:
            mutation_df.to_excel(writer, sheet_name="突变报告", index=False)
        else:
            pd.DataFrame(columns=['结束时间', '起始时间', '币种', '涨跌幅(%)', '当前价格', '当期市值']).to_excel(
                writer, sheet_name="突变报告", index=False
            )

    # 3. 相关性聚类与绘图
    returns_df = pd.DataFrame(returns_dict).dropna()
    corr_matrix = returns_df.corr(method='spearman')

    if len(corr_matrix) > 1:
        dist = pdist(corr_matrix)
        link = linkage(dist, method='ward')
        order = leaves_list(link)
        corr_matrix = corr_matrix.iloc[order, order]

    y_labels = []
    for col in corr_matrix.index:
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

    print("\n分析完成！")
    print(f"1. 已生成统计概览: {overview_file}")
    print(f"2. 已生成突变报告: {mutation_file}")
    if pd.notna(overall_corr_coef):
        print(f"总体相关系数（涨跌幅 vs 交易量/市值）：{overall_corr_coef:.4f}")
        print(f"总体p值：{overall_corr_p:.4g}")
        print(f"总体结论：{overall_corr_text}")
    else:
        print("总体结论：无足够数据计算相关性。")

    plt.show()


# 执行
if __name__ == "__main__":
    # 确保 data 文件夹存在并放入了您的 CSV
    process_crypto_analysis(
        'E:\\py',
        window_days=3,
        mutation_threshold=0.15,
        analysis_start_date='2026-01-01',
        analysis_end_date='2026-04-01'
    )
