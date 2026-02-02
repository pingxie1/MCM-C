import pandas as pd
import numpy as np
from scipy.stats import rankdata
from tqdm import tqdm
import scipy.stats as stats
import re
import itertools
np.random.seed(2026)
def load_data(excel_path='Data_final.xlsx'):
    print("[进度 1/5] 开始加载数据...")
    raw_dataframe = pd.read_excel(excel_path)
    raw_dataframe = raw_dataframe.astype({'id': int, 'season': int, 'Elimination_Week': int, 'placement': int})
    contestant_id_to_name = raw_dataframe.set_index('id')['celebrity_name'].to_dict()
    season_week_data = {}
    max_week_number = 12
    for season in sorted(raw_dataframe['season'].unique()):
        season_dataframe = raw_dataframe[raw_dataframe['season'] == season]
        max_elimination_week = season_dataframe['Elimination_Week'].max()
        season_week_data[season] = {}
        for week in range(1, max_elimination_week + 1):
            active_contestants = season_dataframe[season_dataframe['Elimination_Week'] >= week]
            eliminated_contestants = season_dataframe[season_dataframe['Elimination_Week'] == week]
            active_contestant_ids = active_contestants['id'].tolist()
            if len(active_contestant_ids) < 2:
                continue
            weekly_scores = []
            for contestant_id in active_contestant_ids:
                contestant_row = raw_dataframe[raw_dataframe['id'] == contestant_id].iloc[0]
                score_column = f'week{week}_total_score'
                score_value = contestant_row[score_column] if score_column in raw_dataframe.columns and pd.notna(contestant_row[score_column]) else 0
                weekly_scores.append(score_value)
            score_ranks = rankdata(-np.array(weekly_scores), method='average')
            judge_points = len(active_contestant_ids) + 1 - score_ranks
            judge_points_dict = dict(zip(active_contestant_ids, judge_points))
            judge_scores_dict = dict(zip(active_contestant_ids, weekly_scores))
            season_week_data[season][week] = {
                'active_ids': active_contestant_ids,
                'eliminated_ids': eliminated_contestants['id'].tolist(),
                'n_active': len(active_contestant_ids),
                'n_elim': len(eliminated_contestants),
                'judge_points': judge_points_dict,
                'judge_scores': judge_scores_dict
            }
    print("[进度 1/5] 数据加载完成")
    return raw_dataframe, season_week_data, contestant_id_to_name
def preprocess_data(raw_dataframe):
    print("[进度 2/5] 预处理数据...")
    processed_dataframe = raw_dataframe.copy()
    processed_dataframe['elimination_week'] = processed_dataframe['Elimination_Week']
    score_columns = [col for col in processed_dataframe.columns if 'week' in col and 'total_score' in col]
    week_numbers = set()
    for col in score_columns:
        match = re.search(r'week(\d+)', col)
        if match:
            week_numbers.add(int(match.group(1)))
    max_week_number = max(week_numbers) if week_numbers else 12
    print(f"检测到最大周数为: {max_week_number}")
    long_format_data = []
    for index, row in processed_dataframe.iterrows():
        for week in range(1, max_week_number + 1):
            if week > row['elimination_week']:
                continue
            week_score_columns = [col for col in score_columns if f'week{week}_' in col]
            score_values = pd.to_numeric(row[week_score_columns], errors='coerce')
            if score_values.isna().all():
                continue 
            average_score = score_values.mean()
            if average_score == 0 and week != row['elimination_week']:
                continue
            long_format_data.append({
                'season': row['season'],
                'week': week,
                'celebrity': row['celebrity_name'],
                'avg_judge_score': average_score,
                'elim_week': row['elimination_week']
            })
    print("[进度 2/5] 数据预处理完成")
    return pd.DataFrame(long_format_data)
def calculate_entropy(rank_matrix):
    num_samples, num_contestants = rank_matrix.shape
    entropy_values = []
    for contestant_idx in range(num_contestants):
        contestant_ranks = rank_matrix[:, contestant_idx]
        rank_distribution = pd.Series(contestant_ranks).value_counts(normalize=True)
        entropy = stats.entropy(rank_distribution.values)
        entropy_values.append(entropy)
    return np.array(entropy_values)
def simulate_week(season, week, week_data, prior_dict=None, num_simulations=10000, momentum=50.0, pop_weight=100.0, total_votes_m=10, 
                  momentum_percent=50.0, momentum_rank=0.8, rule_type=None):
    global contestant_id_to_name
    active_contestant_ids = week_data['active_ids']
    num_active_contestants = week_data['n_active']
    num_eliminations = week_data['n_elim']
    eliminated_indices = np.array([active_contestant_ids.index(contestant_id) for contestant_id in week_data['eliminated_ids'] if contestant_id in active_contestant_ids])
    judge_points = np.array([week_data['judge_points'][contestant_id] for contestant_id in active_contestant_ids])
    judge_scores = np.array([week_data['judge_scores'][contestant_id] for contestant_id in active_contestant_ids])
    contestant_names = [contestant_id_to_name.get(contestant_id, 'Unknown') for contestant_id in active_contestant_ids]
    if rule_type is None:
        rule_type = 'Percentage' if 3 <= season <= 27 else 'Rank'
    has_prior = prior_dict is not None
    if prior_dict is None:
        if rule_type == 'Percentage':
            prior_distribution = np.ones(num_active_contestants) / num_active_contestants
        else:
            prior_distribution = np.ones(num_active_contestants) * (num_active_contestants + 1) / 2
    else:
        prior_distribution = np.array([prior_dict.get(contestant_id, (1.0 / num_active_contestants if rule_type == 'Percentage' else (num_active_contestants + 1) / 2)) for contestant_id in active_contestant_ids])
    if rule_type == 'Percentage':
        judge_component = judge_scores / (np.sum(judge_scores) + 1e-8)
        alpha_parameters = 1.0 + momentum_percent * prior_distribution if has_prior else np.ones(num_active_contestants)
        fan_votes_batch = np.random.dirichlet(alpha_parameters, num_simulations)
        total_scores = judge_component + fan_votes_batch
        if num_eliminations > 0:
            kth_index = num_eliminations - 1 if num_eliminations > 0 else 0
            bottom_indices = np.argpartition(total_scores, kth_index, axis=1)[:, :num_eliminations]
            bottom_indices_sorted = np.sort(bottom_indices, axis=1)
            target_indices_sorted = np.sort(eliminated_indices)
            matches = np.all(bottom_indices_sorted == target_indices_sorted, axis=1)
            valid_samples = fan_votes_batch[matches]
        else:
            valid_samples = fan_votes_batch
    elif rule_type == 'Rank':
        judge_component = stats.rankdata(-judge_scores, method='average')
        random_values = np.random.rand(num_simulations, num_active_contestants)
        if has_prior:
            max_prior = np.max(prior_distribution)
            strength = 1.0 - (prior_distribution - 1) / max(max_prior, 1)
            random_values += momentum_rank * strength
        fan_votes_batch = np.argsort(np.argsort(-random_values, axis=1), axis=1) + 1
        total_scores = judge_component + fan_votes_batch
        if num_eliminations > 0:
            kth_index = num_eliminations - 1 if num_eliminations > 0 else 0
            top_indices = np.argpartition(-total_scores, kth_index, axis=1)[:, :num_eliminations]
            top_indices_sorted = np.sort(top_indices, axis=1)
            target_indices_sorted = np.sort(eliminated_indices)
            matches = np.all(top_indices_sorted == target_indices_sorted, axis=1)
            valid_samples = fan_votes_batch[matches]
        else:
            valid_samples = fan_votes_batch
    acceptance_rate = len(valid_samples) / num_simulations if num_simulations > 0 else 0.0
    if len(valid_samples) > 0:
        mean_distribution = valid_samples.mean(axis=0)
        std_distribution = valid_samples.std(axis=0)
        lower_bound = np.percentile(valid_samples, 2.5, axis=0)
        upper_bound = np.percentile(valid_samples, 97.5, axis=0)
        confidence_interval_width = upper_bound - lower_bound
        entropy_values = calculate_entropy(valid_samples) if rule_type == 'Rank' else np.full(num_active_contestants, np.nan)
    else:
        mean_distribution = prior_distribution
        std_distribution = np.zeros(num_active_contestants)
        confidence_interval_width = np.zeros(num_active_contestants)
        entropy_values = np.full(num_active_contestants, np.nan)
    is_validated = False
    if len(valid_samples) > 0:
        if num_eliminations > 0:
            if rule_type == 'Percentage':
                validation_total = judge_component + mean_distribution
                validation_bottom_indices = np.argsort(validation_total)[:num_eliminations]
                is_validated = set(validation_bottom_indices) == set(eliminated_indices)
            elif rule_type == 'Rank':
                validation_total = judge_component + mean_distribution
                validation_top_indices = np.argsort(-validation_total)[:num_eliminations]
                is_validated = set(validation_top_indices) == set(eliminated_indices)
        else:
            is_validated = True
    relative_uncertainty = std_distribution / (mean_distribution + 1e-8)
    simulation_results = []
    new_prior_distribution = {}
    if rule_type == 'Rank':
        popularity_scores = num_active_contestants + 1 - mean_distribution
        total_popularity = np.sum(popularity_scores)
        assumed_proportions = popularity_scores / total_popularity if total_popularity > 0 else np.ones(num_active_contestants) / num_active_contestants
    for index, contestant_id in enumerate(active_contestant_ids):
        if rule_type == 'Percentage':
            estimated_votes = mean_distribution[index] * total_votes_m
            assumed_proportion = np.nan
        else:
            assumed_proportion = assumed_proportions[index]
            estimated_votes = assumed_proportion * total_votes_m
        simulation_results.append({
            'season': season,
            'week': week,
            'id': contestant_id,
            'name': contestant_id_to_name.get(contestant_id, 'Unknown'),
            'mean_vote_prop': float(mean_distribution[index]) if rule_type == 'Percentage' else float(assumed_proportion),
            'mean_rank': float(mean_distribution[index]) if rule_type == 'Rank' else np.nan,
            'estimated_votes_m': float(estimated_votes),
            'vote_std': float(std_distribution[index]),
            'vote_rel_unc': float(relative_uncertainty[index]),
            'Acceptance_Rate': acceptance_rate,
            'CI_Width': float(confidence_interval_width[index]),
            'Entropy': float(entropy_values[index]),
            'Method': rule_type,
            'Validation_By_Mean': is_validated  
        })
        new_prior_distribution[contestant_id] = mean_distribution[index]
    return simulation_results, new_prior_distribution
def unified_evaluation(estimates_dataframe, season_week_data, model_name="Model"):
    rank_matches = []
    percentage_matches = []
    relative_uncertainties = []
    total_weeks_evaluated = 0
    for season in season_week_data:
        for week in season_week_data[season]:
            week_data = season_week_data[season][week]
            if week_data['n_elim'] == 0:
                continue
            total_weeks_evaluated += 1
            week_estimates = estimates_dataframe[(estimates_dataframe['season'] == season) & (estimates_dataframe['week'] == week)]
            if len(week_estimates) != week_data['n_active']:
                continue
            fan_proportions = week_estimates.set_index('id')['mean_vote_prop'].reindex(week_data['active_ids']).values
            judge_points = np.array([week_data['judge_points'][contestant_id] for contestant_id in week_data['active_ids']])
            true_eliminated_set = set(week_data['eliminated_ids'])
            fan_ranks = rankdata(-fan_proportions, method='average')
            judge_ranks = rankdata(-judge_points, method='average')
            total_ranks = fan_ranks + judge_ranks
            predicted_eliminated_set_rank = set(np.array(week_data['active_ids'])[np.argsort(total_ranks)[:week_data['n_elim']]])
            rank_matches.append(predicted_eliminated_set_rank == true_eliminated_set)
            fan_percentages = fan_proportions / (fan_proportions.sum() + 1e-8)
            judge_normalized = judge_points / (judge_points.max() + 1e-8)
            combined_percentages = 0.5 * fan_percentages + 0.5 * judge_normalized
            predicted_eliminated_set_percentage = set(np.array(week_data['active_ids'])[np.argsort(combined_percentages)[:week_data['n_elim']]])
            percentage_matches.append(predicted_eliminated_set_percentage == true_eliminated_set)
            relative_uncertainties.extend(week_estimates['vote_rel_unc'].values)
    print(f"\n=== {model_name} 统一评测结果 ===")
    print(f"PPC 排名制精确匹配率: {np.mean(rank_matches):.3f} (周数: {len(rank_matches)})")
    print(f"PPC 百分比制精确匹配率: {np.mean(percentage_matches):.3f} (周数: {len(percentage_matches)})")
    print(f"整体平均相对不确定性: {np.mean(relative_uncertainties):.3f}")
    print(f"总评测周数: {total_weeks_evaluated}")
def run_simulation_with_params(long_format_data, season_week_data, momentum_percent, momentum_rank, num_simulations=2000, progress=False):
    all_simulation_results = []
    seasons = sorted(long_format_data['season'].unique())
    iterator = tqdm(seasons, desc="Simulating") if progress else seasons
    for season in iterator:
        prior_distribution = None
        weeks = sorted(season_week_data[season].keys())
        rule_type = 'Percentage' if 3 <= season <= 27 else 'Rank'
        for week in weeks:
            week_data = season_week_data[season][week]
            week_simulation_results, prior_distribution = simulate_week(
                season, week, week_data, 
                prior_dict=prior_distribution, 
                num_simulations=num_simulations,
                momentum_percent=momentum_percent,
                momentum_rank=momentum_rank,
                rule_type=rule_type
            )
            all_simulation_results.extend(week_simulation_results)
    return pd.DataFrame(all_simulation_results)
def main():
    global contestant_id_to_name
    raw_dataframe, season_week_data, contestant_id_to_name = load_data()
    long_format_data = preprocess_data(raw_dataframe)
    if long_format_data.empty: return
    print("[进度 3/5] 开始网格搜索优化参数...")
    possible_percent_parameters = list(range(0, 101, 10)) 
    possible_rank_parameters = [round(x * 0.1, 1) for x in range(0, 21)]
    print(f"Percentage Candidates: {possible_percent_parameters}")
    print(f"Rank Candidates: {possible_rank_parameters}")
    parameter_combinations = list(itertools.product(possible_percent_parameters, possible_rank_parameters))
    best_validation_score = -1
    best_parameters = (50.0, 0.8)
    SEARCH_NUM_SIMULATIONS = 1000
    for percent_param, rank_param in tqdm(parameter_combinations, desc="Grid Searching"):
        simulation_results = run_simulation_with_params(long_format_data, season_week_data, momentum_percent=percent_param, momentum_rank=rank_param, num_simulations=SEARCH_NUM_SIMULATIONS)
        if not simulation_results.empty:
            weekly_results = simulation_results.drop_duplicates(['season', 'week'])
            validation_score = weekly_results['Validation_By_Mean'].mean()
        else:
            validation_score = 0
        if validation_score > best_validation_score:
            best_validation_score = validation_score
            best_parameters = (percent_param, rank_param)
    print("[进度 3/5] 网格搜索完成")
    print(f"\n最优参数已找到！(Target: Maximize Validation Accuracy)")
    print(f"MOMENTUM_PERCENT (For Season 3-27): {best_parameters[0]}")
    print(f"MOMENTUM_RANK (For Season 1-2, 28+): {best_parameters[1]}")
    print(f"预计平均接受率: {best_validation_score:.2%}")
    print("[进度 4/5] 使用最优参数运行最终高精度模拟...")
    final_simulation_results = run_simulation_with_params(
        long_format_data, season_week_data, 
        momentum_percent=best_parameters[0], 
        momentum_rank=best_parameters[1], 
        num_simulations=10000, 
        progress=True
    )
    print("[进度 5/5] 生成投票估计...")
    final_simulation_results.to_csv('mcrif_estimates5.csv', index=False)
    print("投票估计完成，保存至 mcrif_estimates5.csv")
    unified_evaluation(final_simulation_results, season_week_data, "MCRIF")
    if not final_simulation_results.empty:
        average_acceptance_rate = final_simulation_results['Acceptance_Rate'].mean()
        print(f"最终平均接受率: {average_acceptance_rate:.2%}")
if __name__ == "__main__":
    main()