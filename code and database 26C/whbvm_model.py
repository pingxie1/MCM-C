import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from scipy.stats import rankdata
import warnings
warnings.filterwarnings("ignore")
print("[进度 1/6] 开始加载数据...")
df = pd.read_excel('Data_final.xlsx')
df = df.astype({'id': int, 'season': int, 'Elimination_Week': int, 'placement': int})
seasons = sorted(df['season'].unique())
player_ids = sorted(df['id'].unique())
n_players = len(player_ids)
id_to_idx = {pid: i for i, pid in enumerate(player_ids)}
id_to_pop = df.set_index('id')['popularity_i'].fillna(0).to_dict()
id_to_name = df.set_index('id')['celebrity_name'].to_dict()
season_to_maxw = {}
max_week = 12
for s in seasons:
    s_df = df[df['season'] == s]
    maxw = s_df['Elimination_Week'].max()
    season_to_maxw[s] = maxw
print("[进度 1/6] 数据加载完成")
print("[进度 2/6] 预处理标准化和积分...")
tilde_J_dict = {}
tilde_H_dict = {}
judge_points_dict = {}
raw_scores = np.full((n_players, max_week + 1), np.nan)
raw_H = np.zeros((n_players, max_week + 1))
for pid in player_ids:
    idx = id_to_idx[pid]
    row = df[df['id'] == pid].iloc[0]
    for w in range(1, max_week + 1):
        col = f'week{w}_total_score'
        if col in df.columns and pd.notna(row[col]):
            raw_scores[idx, w] = row[col]
for idx in range(n_players):
    for w in range(2, max_week + 1):
        prev = raw_scores[idx, 1:w]
        valid = prev[~np.isnan(prev)]
        raw_H[idx, w] = np.mean(valid) if len(valid) > 0 else 0.0
for s in seasons:
    for w in range(1, season_to_maxw[s] + 1):
        active_df = df[(df['season'] == s) & (df['Elimination_Week'] >= w)]
        active_idxs = [id_to_idx[iid] for iid in active_df['id']]
        n_active = len(active_idxs)
        j_vals = raw_scores[active_idxs, w]
        h_vals = raw_H[active_idxs, w]
        mu_j, std_j = np.nanmean(j_vals), np.nanstd(j_vals)
        std_j = std_j if std_j > 0 else 1.0
        mu_h, std_h = np.mean(h_vals), np.std(h_vals)
        std_h = std_h if std_h > 0 else 1.0
        j_vals_filled = np.nan_to_num(j_vals, nan=np.nanmin(j_vals[~np.isnan(j_vals)]) - 1 if np.any(~np.isnan(j_vals)) else 0)
        ranks = rankdata(-j_vals_filled, method='average')
        points = n_active + 1 - ranks
        for a_i, p_idx in enumerate(active_idxs):
            pid = active_df['id'].iloc[a_i]
            tilde_J_dict[(s, w, p_idx)] = (raw_scores[p_idx, w] - mu_j) / std_j if not np.isnan(raw_scores[p_idx, w]) else 0.0
            tilde_H_dict[(s, w, p_idx)] = (raw_H[p_idx, w] - mu_h) / std_h
            judge_points_dict[(s, w, p_idx)] = points[a_i]
print("[进度 2/6] 预处理完成")
print("[进度 3/6] 构建观测...")
season_to_elim_obs = {}
season_to_week_data = {}
observed_dict = {}
for s in seasons:
    s_df = df[df['season'] == s]
    maxw = season_to_maxw[s]
    elim_obs = []
    week_data = {}
    for w in range(1, maxw):
        active_df = s_df[s_df['Elimination_Week'] >= w]
        active_ids = sorted(active_df['id'].tolist())
        if len(active_ids) < 2:
            continue
        eliminated_ids = sorted(s_df[s_df['Elimination_Week'] == w]['id'].tolist())
        n_active = len(active_ids)
        data = {'active_ids': active_ids, 'eliminated_ids': eliminated_ids, 'n_active': n_active}
        points = np.array([judge_points_dict.get((s, w, id_to_idx[iid]), 1.0) for iid in active_ids])
        data['judge_points'] = {active_ids[k]: points[k] for k in range(n_active)}
        week_data[w] = data
        if len(eliminated_ids) == 0:
            continue
        elim_idxs = [active_ids.index(eid) for eid in eliminated_ids if eid in active_ids]
        n_elim = len(eliminated_ids)
        counts = np.zeros(n_active, dtype='int64')
        counts[elim_idxs] = 1
        observed_dict[(s, w)] = counts
        elim_obs.append((w, active_ids, n_active, n_elim))
    season_to_elim_obs[s] = elim_obs
    season_to_week_data[s] = week_data
print("[进度 3/6] 观测构建完成")
print("[进度 4/6] 构建模型并采样...")
with pm.Model() as model:
    μθ = pm.Normal('μθ', mu=0, sigma=10)
    λ = pm.Normal('λ', mu=0, sigma=1.0)
    σθ = pm.HalfNormal('σθ', sigma=2.0)
    βJ = pm.Normal('βJ', mu=0, sigma=1.0)
    βH = pm.Normal('βH', mu=0, sigma=1.0)
    pop_array = np.array([id_to_pop.get(pid, 0.0) for pid in player_ids])
    θ = pm.Normal('θ', mu=μθ + λ * pop_array, sigma=σθ, shape=n_players)
    for s in seasons:
        for w, active_ids, n_active, n_elim in season_to_elim_obs[s]:
            active_idxs = [id_to_idx[iid] for iid in active_ids]
            J_tilde = pt.stack([tilde_J_dict.get((s, w, idx), 0.0) for idx in active_idxs])
            H_tilde = pt.stack([tilde_H_dict.get((s, w, idx), 0.0) for idx in active_idxs])
            F = θ[active_idxs] + βJ * J_tilde + βH * H_tilde
            p_vote = pm.math.softmax(F)
            tau = 0.05
            diffs = p_vote.dimshuffle(0, 'x') - p_vote.dimshuffle('x', 0)
            sig_matrix = pt.sigmoid(-diffs / tau)
            soft_vote_rank = 1 + pt.sum(sig_matrix, axis=1) - 0.5
            soft_vote_points = n_active + 1 - soft_vote_rank
            judge_points = pt.stack([judge_points_dict.get((s, w, idx), 1.0) for idx in active_idxs])
            total_points = soft_vote_points + judge_points
            p_elim = pm.math.softmax(-total_points)
            pm.Multinomial(f'elim_s{s}_w{w}', n=n_elim, p=p_elim, observed=observed_dict[(s, w)])
    idata = pm.sample(
        draws=1000,
        tune=1500,
        target_accept=0.99,
        chains=4,
        nuts_sampler='numpyro',
        random_seed=42,
        init='adapt_diag'
    )
print("[进度 5/6] 采样完成，参数总结：")
print(az.summary(idata, var_names=['μθ', 'λ', 'βJ', 'βH', 'σθ']))
print("[进度 5/6] 生成投票估计...")
post = idata.posterior
θ_samples = post['θ'].stack(sample=('chain', 'draw')).values
βJ_samples = post['βJ'].values.flatten()
βH_samples = post['βH'].values.flatten()
n_samples_total = θ_samples.shape[0]
n_select = min(500, n_samples_total)
selected_indices = np.random.choice(n_samples_total, size=n_select, replace=False)
votes_list = []
for s in seasons:
    for w in range(1, season_to_maxw[s] + 1):
        active_df = df[(df['season'] == s) & (df['Elimination_Week'] >= w)]
        active_ids = active_df['id'].tolist()
        if len(active_ids) < 2:
            continue
        active_idxs = [id_to_idx[iid] for iid in active_ids]
        vote_props = np.zeros((n_select, len(active_ids)))
        for i, sel in enumerate(selected_indices):
            θ_sel = θ_samples[sel, active_idxs]
            J_sel = np.array([tilde_J_dict.get((s, w, idx), 0.0) for idx in active_idxs])
            H_sel = np.array([tilde_H_dict.get((s, w, idx), 0.0) for idx in active_idxs])
            F_sel = θ_sel + βJ_samples[sel] * J_sel + βH_samples[sel] * H_sel
            p_sel = np.exp(F_sel - np.max(F_sel))
            p_sel /= p_sel.sum()
            vote_props[i] = p_sel
        mean_prop = vote_props.mean(axis=0)
        std_prop = vote_props.std(axis=0)
        rel_unc = std_prop / (mean_prop + 1e-8)
        for j, pid in enumerate(active_ids):
            votes_list.append({
                'season': s,
                'week': w,
                'id': pid,
                'name': id_to_name.get(pid, 'Unknown'),
                'mean_vote_prop': mean_prop[j],
                'vote_std': std_prop[j],
                'vote_rel_unc': rel_unc[j]
            })
votes_df = pd.DataFrame(votes_list)
votes_df.to_csv('whbvm_estimates.csv', index=False)
print("投票估计完成，保存至 whbvm_estimates.csv")
def unified_evaluation(votes_df, season_to_week_data, model_name="WHBVM"):
    matches_rank = []
    matches_pct = []
    rel_uncs = []
    total_weeks = 0
    for s in seasons:
        if s not in season_to_week_data:
            continue
        for w, data in season_to_week_data[s].items():
            if len(data['eliminated_ids']) == 0:
                continue
            total_weeks += 1
            sub = votes_df[(votes_df['season'] == s) & (votes_df['week'] == w)]
            if len(sub) != data['n_active']:
                continue
            fan_prop = sub['mean_vote_prop'].values
            judge_pts = np.array([data['judge_points'].get(iid, 1) for iid in data['active_ids']])
            true_set = set(data['eliminated_ids'])
            fan_rank = rankdata(-fan_prop, method='average')
            judge_rank = rankdata(-judge_pts, method='average')
            total_rank = fan_rank + judge_rank
            pred_set_rank = set(np.array(data['active_ids'])[np.argsort(total_rank)[:len(data['eliminated_ids'])]])
            matches_rank.append(pred_set_rank == true_set)
            fan_pct = fan_prop / (fan_prop.sum() + 1e-8)
            judge_norm = judge_pts / (judge_pts.max() + 1e-8)
            combined_pct = 0.5 * fan_pct + 0.5 * judge_norm
            pred_set_pct = set(np.array(data['active_ids'])[np.argsort(combined_pct)[:len(data['eliminated_ids'])]])
            matches_pct.append(pred_set_pct == true_set)
            rel_uncs.extend(sub['vote_rel_unc'].values)
    print(f"\n=== {model_name} 统一评测结果 ===")
    print(f"PPC 排名制精确匹配率: {np.mean(matches_rank):.3f} (周数: {len(matches_rank)})")
    print(f"PPC 百分比制精确匹配率: {np.mean(matches_pct):.3f} (周数: {len(matches_pct)})")
    print(f"整体平均相对不确定性: {np.mean(rel_uncs):.3f}")
    print(f"总评测周数: {total_weeks}")
unified_evaluation(votes_df, season_to_week_data, "WHBVM")
print("\n所有任务与结果验证完成！")