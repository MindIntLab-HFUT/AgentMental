import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, 
    r2_score, 
    f1_score, 
    accuracy_score,
    cohen_kappa_score  
)
from pingouin import intraclass_corr
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns


df_real = pd.read_csv('./data/depression_new_111.csv')

df_pred = pd.read_csv('/final_result/72b-32b.csv')  


# 数据对齐和清洗
common_videos = set(df_real['video_name']).intersection(df_pred['video_name'])
df_real = df_real[df_real['video_name'].isin(common_videos)].sort_values('video_name').reset_index(drop=True)
df_pred = df_pred[df_pred['video_name'].isin(common_videos)].sort_values('video_name').reset_index(drop=True)

assert df_real['video_name'].equals(df_pred['video_name']), "数据未对齐!"


def calculate_icc(data, targets, raters, ratings):
    """带异常处理的ICC计算"""
    try:
        icc = intraclass_corr(
            data=data,
            targets=targets,
            raters=raters,
            ratings=ratings
        )
        return icc.loc[icc['Type'] == 'ICC3k', 'ICC'].values[0]
    except Exception as e:
        print(f"ICC计算错误: {str(e)}")
        return np.nan


total_metrics = {
    'MAE': mean_absolute_error(df_real['total'], df_pred['total']),
    # 'R²': r2_score(df_real['total'], df_pred['total']),
    'Pearson': pearsonr(df_real['total'], df_pred['total'])[0],
    'ICC(3,k)': calculate_icc(
        pd.DataFrame({
            'video_name': df_real['video_name'].tolist() * 2,
            'rater': ['real']*len(df_real) + ['pred']*len(df_pred),
            'score': np.concatenate([df_real['total'], df_pred['total']])
        }),
        targets='video_name',
        raters='rater',
        ratings='score'
    )
}


category_metrics = {
    'Accuracy': accuracy_score(df_real['classes'], df_pred['classes']),
    'Macro_F1': f1_score(df_real['classes'], df_pred['classes'], average='macro'),
    'Micro_F1': f1_score(df_real['classes'], df_pred['classes'], average='micro'),
    'Kappa': cohen_kappa_score(df_real['classes'], df_pred['classes'])  # 新增Kappa

}

item_scoring = {
    "HAMD-17": {
        "抑郁情绪": 5,
        "有罪感": 5,
        "自杀": 5,
        "入睡困难": 3,
        "睡眠不深": 3,
        "早醒": 3,
        "工作和兴趣": 5,
        "迟缓": 5,
        "激越": 5,
        "精神性焦虑": 5,
        "躯体性焦虑": 5,
        "胃肠道症状": 3,
        "全身症状": 3,
        "性症状": 3,
        "疑病": 5,
        "体重减轻": 3,
        "自知力": 3
    }
}

items = [f'item{i}' for i in range(1, 18)]
item_names = list(item_scoring["HAMD-17"].keys())
vertical_results = []

for i, item in enumerate(items):
    real_scores = df_real[item]
    pred_scores = df_pred[item]
    
    if real_scores.sum() == 0 and pred_scores.sum() == 0:
        continue

    num_classes = item_scoring["HAMD-17"][item_names[i]]
    accuracy = accuracy_score(real_scores, pred_scores)

    vertical_results.append({
        'Item': item,
        'MAE': mean_absolute_error(real_scores, pred_scores),
        # 'R²': r2_score(real_scores, pred_scores),
        'ICC(3,k)': calculate_icc(
            pd.DataFrame({
                'video_name': df_real['video_name'].tolist() * 2,
                'rater': ['real']*len(df_real) + ['pred']*len(df_pred),
                'score': np.concatenate([real_scores, pred_scores])
            }),
            targets='video_name',
            raters='rater',
            ratings='score'
        ),
        'Pearson': pearsonr(real_scores, pred_scores)[0],
        'Accuracy': accuracy
    })

total_df = pd.DataFrame([total_metrics])
category_df = pd.DataFrame([category_metrics])
items_df = pd.DataFrame(vertical_results)


print("\n=== 横向评估（总分） ===")
print(total_df.round(3))

print("\n=== 症状分类评估 ===")
print(category_df.round(3))

print("\n=== 项目级评估 ===")
print(items_df.round(3))
