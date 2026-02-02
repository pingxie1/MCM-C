import pandas as pd
import numpy as np

# 1. 加载数据
# 注意：文件名包含中文字符，在某些环境中可能需要确保编码正确
data = pd.read_excel('2026_MCM_Problem_C_Sen_NA版_long_format_sorted_with_fan_votes.xlsx')

# 2. 数据清理
# 确保必要列存在
required_columns = ['season', 'week', 'celebrity_name', 'judges_score', 'fan_votes', 'results']
for col in required_columns:
    if col not in data.columns:
        data[col] = np.nan

# 将judges_score和fan_votes转换为float，处理空值
data['judges_score'] = pd.to_numeric(data['judges_score'], errors='coerce')
data['fan_votes'] = pd.to_numeric(data['fan_votes'], errors='coerce')
data['season'] = pd.to_numeric(data['season'], errors='coerce')
data['week'] = pd.to_numeric(data['week'], errors='coerce')

# 移除无效行（赛季或周次为空）
data = data.dropna(subset=['season', 'week'])

# 确保赛季和周次为整数
data['season'] = data['season'].astype(int)
data['week'] = data['week'].astype(int)

# 3. 定义两种投票合并方法函数
def rank_method(scores, fan_votes_estimated):
    # 计算评审排名：最高分排名1
    judge_ranks = scores.rank(ascending=False, method='min')
    # 计算粉丝排名：最高投票排名1
    fan_ranks = fan_votes_estimated.rank(ascending=False, method='min')
    # 合并排名：总排名值最小者最好，最大者最差（被淘汰）
    total_ranks = judge_ranks + fan_ranks
    return total_ranks

def percentage_method(scores, fan_votes_estimated):
    # 计算评审百分比
    judge_percent = (scores / scores.sum()) * 100 if scores.sum() > 0 else pd.Series(0, index=scores.index)
    # 计算粉丝百分比（假设fan_votes_estimated是比例或相对值，先归一化）
    fan_sum = fan_votes_estimated.sum()
    fan_percent = (fan_votes_estimated / fan_sum) * 100 if fan_sum > 0 else pd.Series(0, index=fan_votes_estimated.index)
    # 合并百分比：总百分比最低者被淘汰
    total_percent = judge_percent + fan_percent
    return total_percent

# 4. 估计粉丝投票数
def estimate_fan_votes(week_data, add_noise=False, noise_std=0.01):
    """估计粉丝投票数，可选择添加高斯噪声模拟不确定性"""
    base_votes = week_data['fan_votes']
    if add_noise:
        # 添加高斯噪声
        noise = np.random.normal(0, noise_std, size=len(base_votes))
        noisy_votes = base_votes + noise
        # 确保值非负
        noisy_votes = np.maximum(0, noisy_votes)
        # 归一化
        total = noisy_votes.sum()
        if total > 0:
            noisy_votes = noisy_votes / total
        return pd.Series(noisy_votes, index=base_votes.index)
    else:
        return base_votes

# 5. 根据赛季确定使用的方法
def get_method_for_season(season):
    """根据赛季编号返回应该使用的方法"""
    if season in [1, 2] or (season >= 28 and season <= 34):
        return 'rank_method'
    elif season >= 3 and season <= 27:
        return 'percentage_method'
    else:
        return 'rank_method'  # 默认使用排名法

# 6. 应用方法并找出矛盾
contradictory_list = []
# 已知的争议选手
controversial_contestants = {
    'Jerry Rice': 2,
    'Billy Ray Cyrus': 4,
    'Bristol Palin': 11,
    'Bobby Bones': 27
}

for season in sorted(data['season'].unique()):
    season_data = data[data['season'] == season]
    for week in sorted(season_data['week'].dropna().unique()):
        week_data = season_data[season_data['week'] == week].dropna(subset=['judges_score', 'fan_votes'])
        if len(week_data) < 2:
            continue  # 如果少于2位选手，跳过（无淘汰）
        
        scores = week_data['judges_score']
        fan_votes_estimated = estimate_fan_votes(week_data)
        
        # 应用排名法
        rank_total = rank_method(scores, fan_votes_estimated)
        eliminated_rank_idx = rank_total.idxmax()  # 总排名最大（最差）
        eliminated_rank_name = week_data.loc[eliminated_rank_idx, 'celebrity_name']
        
        # 应用百分比法
        percent_total = percentage_method(scores, fan_votes_estimated)
        eliminated_percent_idx = percent_total.idxmin()  # 总百分比最小（最差）
        eliminated_percent_name = week_data.loc[eliminated_percent_idx, 'celebrity_name']
        
        # 确定该赛季应使用的方法
        actual_method = get_method_for_season(season)
        
        # 查找实际结果（假设实际淘汰的是results中包含'Eliminated'或'Week {week}'的）
        actual_eliminated = week_data[week_data['results'].astype(str).str.contains('Eliminated|Week ' + str(int(week)), na=False)]['celebrity_name'].values
        actual_result = actual_eliminated[0] if len(actual_eliminated) > 0 else 'Unknown'
        
        # 检查是否为争议选手
        is_controversial = False
        for contestant, contestant_season in controversial_contestants.items():
            if contestant_season == season and any(week_data['celebrity_name'].str.contains(contestant, na=False)):
                is_controversial = True
                break
        
        # 如果淘汰者不同，则记录矛盾
        if eliminated_rank_idx != eliminated_percent_idx:
            contradictory_list.append({
                'season': season,
                'week': week,
                'celebrity_name_rank': eliminated_rank_name,
                'celebrity_name_percent': eliminated_percent_name,
                'rank_method_eliminated': eliminated_rank_name,
                'percent_method_eliminated': eliminated_percent_name,
                'actual_eliminated': actual_result,
                'actual_method': actual_method,
                'is_controversial': is_controversial
            })

# 6. 输出矛盾选手文件
contradictory_df = pd.DataFrame(contradictory_list)
contradictory_df.to_csv('contradictory_contestants.csv', index=False)

# 7. 应用第三个规则（评委淘汰选择）
def predict_judge_choice(bottom_two):
    """预测评委的淘汰选择
    基于评审分数：选择评审分数最低的选手
    可扩展为机器学习模型，使用更多特征
    """
    if len(bottom_two) < 2:
        return 'Unknown'
    
    # 基于评审分数选择最低分
    min_score_idx = bottom_two['judges_score'].idxmin()
    judge_eliminated = bottom_two.loc[min_score_idx, 'celebrity_name']
    
    return judge_eliminated

judge_prediction_list = []
for index, row in contradictory_df.iterrows():
    season = row['season']
    week = row['week']
    week_data = data[(data['season'] == season) & (data['week'] == week)].dropna(subset=['judges_score'])
    
    # 底二选手：rank_method和percent_method淘汰的两人
    bottom_two_names = [row['rank_method_eliminated'], row['percent_method_eliminated']]
    bottom_two = week_data[week_data['celebrity_name'].isin(bottom_two_names)]
    
    # 应用评委选择规则
    judge_eliminated = predict_judge_choice(bottom_two)
    
    judge_prediction_list.append({
        'season': season,
        'week': week,
        'rank_method_eliminated': row['rank_method_eliminated'],
        'percent_method_eliminated': row['percent_method_eliminated'],
        'judge_prediction_eliminated': judge_eliminated
    })

# 8. 输出评委预测结果
judge_prediction_df = pd.DataFrame(judge_prediction_list)
judge_prediction_df.to_csv('judge_prediction_results.csv', index=False)

# 9. 生成分析报告
# 运行蒙特卡洛模拟，计算不确定性
print("开始运行蒙特卡洛模拟，分析不确定性...")
num_simulations = 100
contradiction_counts = []

# 选择一个示例周进行模拟
if len(data) > 0:
    sample_season = sorted(data['season'].unique())[0]
    sample_week = sorted(data[data['season'] == sample_season]['week'].unique())[0]
    sample_data = data[(data['season'] == sample_season) & (data['week'] == sample_week)].dropna(subset=['judges_score', 'fan_votes'])
    
    if len(sample_data) >= 2:
        for _ in range(num_simulations):
            scores = sample_data['judges_score']
            fan_votes_estimated = estimate_fan_votes(sample_data, add_noise=True)
            
            # 应用两种方法
            rank_total = rank_method(scores, fan_votes_estimated)
            eliminated_rank_idx = rank_total.idxmax()
            
            percent_total = percentage_method(scores, fan_votes_estimated)
            eliminated_percent_idx = percent_total.idxmin()
            
            # 记录是否矛盾
            contradiction_counts.append(1 if eliminated_rank_idx != eliminated_percent_idx else 0)

# 计算统计结果
if contradiction_counts:
    contradiction_rate = sum(contradiction_counts) / num_simulations
    # 计算95%置信区间（使用二项分布近似）
    import math
    se = math.sqrt(contradiction_rate * (1 - contradiction_rate) / num_simulations)
    ci_lower = max(0, contradiction_rate - 1.96 * se)
    ci_upper = min(1, contradiction_rate + 1.96 * se)
else:
    contradiction_rate = 0
    ci_lower, ci_upper = 0, 0

# 统计矛盾选手情况
num_contradictions = len(contradictory_df)
num_controversial = len(contradictory_df[contradictory_df['is_controversial'] == True])