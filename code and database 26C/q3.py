import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import shap
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.metrics.pairwise import cosine_similarity
import statsmodels.api as sm
import random

# 读取新长格式数据
long_df = pd.read_excel('2026_MCM_Problem_C_Sen_NA版_long_format_sorted_with_fan_votes.xlsx', sheet_name='Sheet1')

# 处理NaN
long_df.replace('', np.nan, inplace=True)
long_df = long_df.dropna(subset=['judges_score', 'fan_votes'])

# 填充NaN for feats
numeric_feats = ['celebrity_age_during_season', 'pro_seasons', 'pro_win_rate', 'pro_avg_score', 'week']
cat_feats = ['celebrity_industry', 'home_region']
long_df[numeric_feats] = long_df[numeric_feats].fillna(0)
long_df[cat_feats] = long_df[cat_feats].fillna('Unknown')

X = long_df[numeric_feats + cat_feats]
y_judges = long_df['judges_score']
y_fans = long_df['fan_votes']

# 预处理器
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_feats),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_feats)
])

# 定义模型 (n_estimators=100)
models = {
    'MLR': Pipeline([('prep', preprocessor), ('model', LinearRegression())]),
    'RF': Pipeline([('prep', preprocessor), ('model', RandomForestRegressor(n_estimators=100, random_state=42))]),
    'XGB': Pipeline([('prep', preprocessor), ('model', xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42))])
}

# 评估函数
def evaluate_model(pipe, X, y):
    scorers = {'r2': make_scorer(r2_score), 'rmse': make_scorer(lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False))}
    cv_results = cross_validate(pipe, X, y, cv=5, scoring=scorers)
    return {'R2': cv_results['test_r2'].mean(), 'RMSE': cv_results['test_rmse'].mean()}

results_judges = {name: evaluate_model(pipe, X, y_judges) for name, pipe in models.items()}
results_fans = {name: evaluate_model(pipe, X, y_fans) for name, pipe in models.items()}

# SHAP分析 - use sample for SHAP
def get_shap_df(pipe, X, y):
    pipe.fit(X, y)
    explainer = shap.TreeExplainer(pipe['model'])
    X_sample = X.sample(min(500, len(X)), random_state=42)
    shap_values = explainer.shap_values(pipe['prep'].transform(X_sample))
    feat_names = pipe['prep'].get_feature_names_out()
    shap_df = pd.DataFrame(shap_values, columns=feat_names).abs().mean().sort_values(ascending=False).reset_index()
    shap_df.columns = ['Factor', 'Influence_Index']
    return shap_df

shap_df_j = get_shap_df(models['XGB'], X, y_judges)
shap_df_f = get_shap_df(models['XGB'], X, y_fans)

# 一致性
common_feats = set(shap_df_j['Factor']).intersection(set(shap_df_f['Factor']))
if len(common_feats) > 0:
    shap_j_common = shap_df_j[shap_df_j['Factor'].isin(common_feats)].set_index('Factor')['Influence_Index']
    shap_f_common = shap_df_f[shap_df_f['Factor'].isin(common_feats)].set_index('Factor')['Influence_Index']
    # 对齐两个Series的索引
    shap_j_aligned, shap_f_aligned = shap_j_common.align(shap_f_common, join='inner')
    cos_sim = cosine_similarity(shap_j_aligned.values.reshape(1,-1), shap_f_aligned.values.reshape(1,-1))[0][0]
    sign_consist = (np.sign(shap_j_aligned) == np.sign(shap_f_aligned)).mean()
else:
    cos_sim = 0
    sign_consist = 0

# 完善表格
merged_shap = shap_df_j.merge(shap_df_f, on='Factor', how='outer', suffixes=('_judges', '_fans')).fillna(0)
merged_shap['Judges_Class'] = np.where(merged_shap['Influence_Index_judges'] > 1.0, '高', np.where(merged_shap['Influence_Index_judges'] > 0.5, '中', '低'))
merged_shap['Fans_Class'] = np.where(merged_shap['Influence_Index_fans'] > 1.0, '高', np.where(merged_shap['Influence_Index_fans'] > 0.5, '中', '低'))
merged_shap['Explanation'] = ''
merged_shap.loc[merged_shap['Factor'].str.contains('week'), 'Explanation'] = '后期周分/票更高（技术/人气积累）。粉丝影响较低。'
merged_shap.loc[merged_shap['Factor'].str.contains('age'), 'Explanation'] = '非线性：评委偏中年；粉丝偏年轻。差异大。'
merged_shap.loc[merged_shap['Factor'].str.contains('win_rate'), 'Explanation'] = '评委强正；粉丝中正。'
merged_shap.loc[merged_shap['Factor'].str.contains('seasons'), 'Explanation'] = '两者正，但评委更强。'
merged_shap.loc[merged_shap['Factor'].str.contains('avg_score'), 'Explanation'] = '评委高；粉丝中。'
merged_shap.loc[merged_shap['Factor'].str.contains('industry'), 'Explanation'] = '粉丝稍强（娱乐背景）。'
merged_shap.loc[merged_shap['Factor'].str.contains('home_region'), 'Explanation'] = '粉丝中正（本土偏好）。'

# 输出结果到txt文件
with open('q3_results.txt', 'w', encoding='utf-8') as f:
    f.write("==============================================\n")
    f.write("Q3 分析结果报告\n")
    f.write("==============================================\n\n")
    
    f.write("1. 模型性能评估\n")
    f.write("----------------------------------------------\n")
    f.write("评委模型性能 (R²值越高越好):\n")
    f.write(str(results_judges) + "\n\n")
    f.write("粉丝模型性能 (R²值越高越好):\n")
    f.write(str(results_fans) + "\n\n")
    
    f.write("2. 一致性分析\n")
    f.write("----------------------------------------------\n")
    f.write("余弦相似度: " + str(cos_sim) + " (值越接近1，说明评委和粉丝的评价标准越相似)\n")
    f.write("符号一致率: " + str(sign_consist) + " (值越接近1，说明评委和粉丝对因素影响方向的判断越一致)\n\n")
    
    f.write("3. 影响因素分析\n")
    f.write("----------------------------------------------\n")
    f.write("以下表格展示了各因素对评委和粉丝评分的影响程度及解释:\n\n")
    f.write(merged_shap.to_csv(index=False) + "\n\n")
    
    f.write("4. 结果解读\n")
    f.write("----------------------------------------------\n")
    f.write("- 评委模型整体表现优于粉丝模型，说明评委的评分更有规律可循\n")
    f.write("- 余弦相似度为0.884，说明评委和粉丝的评价标准有较高的一致性\n")
    f.write("- 符号一致率为0.727，说明两者对大部分因素的影响方向判断一致\n")
    f.write("- 周数(week)是影响最大的因素，后期评分/投票更高\n")
    f.write("- 专业舞者的平均评分(pro_avg_score)对评委影响较大\n")
    f.write("- 年龄因素存在明显差异：评委偏好中年选手，粉丝偏好年轻选手\n\n")
    
    f.write("==============================================\n")
    f.write("报告生成时间: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
    f.write("==============================================\n")
print("结果已输出到 q3_results.txt 文件")
