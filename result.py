import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, 
    f1_score, 
    accuracy_score,
    cohen_kappa_score
)
from pingouin import intraclass_corr
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

df_real = pd.read_csv('')
df_pred = pd.read_csv('')

df_real = df_real.rename(columns={
    'Participant_ID': 'identifier',
    'PHQ8_Score': 'total',
    'PHQ8_Binary': 'classes'
})

common_ids = set(df_real['identifier']).intersection(df_pred['identifier'])
df_real = df_real[df_real['identifier'].isin(common_ids)].sort_values('identifier').reset_index(drop=True)
df_pred = df_pred[df_pred['identifier'].isin(common_ids)].sort_values('identifier').reset_index(drop=True)

assert df_real['identifier'].equals(df_pred['identifier']), "Data not aligned!"

def calculate_icc(data, targets, raters, ratings):
    try:
        icc = intraclass_corr(
            data=data,
            targets=targets,
            raters=raters,
            ratings=ratings
        )
        return icc.loc[icc['Type'] == 'ICC3k', 'ICC'].values[0]
    except Exception as e:
        print(f"ICC calculation error: {str(e)}")
        return np.nan

total_metrics = {
    'MAE': mean_absolute_error(df_real['total'], df_pred['total']),
    'Pearson': pearsonr(df_real['total'], df_pred['total'])[0],
    'ICC(3,k)': calculate_icc(
        pd.DataFrame({
            'identifier': df_real['identifier'].tolist() * 2,
            'rater': ['real']*len(df_real) + ['pred']*len(df_pred),
            'score': np.concatenate([df_real['total'], df_pred['total']])
        }),
        targets='identifier',
        raters='rater',
        ratings='score'
    )
}

category_metrics = {
    'Accuracy': accuracy_score(df_real['classes'], df_pred['classes']),
    'Macro_F1': f1_score(df_real['classes'], df_pred['classes'], average='macro'),
    # 'Micro_F1': f1_score(df_real['classes'], df_pred['classes'], average='micro'),
    'F1_Control': f1_score(df_real['classes'], df_pred['classes'], labels=[0], pos_label=0),
    'F1_Depression': f1_score(df_real['classes'], df_pred['classes'], labels=[1], pos_label=1),
    'Kappa': cohen_kappa_score(df_real['classes'], df_pred['classes'])
}

item_scoring = {
    "PHQ-8": {
        "Loss of Interest": 3,
        "Depressed Mood": 3,
        "Sleep Problems": 3,
        "Fatigue or Low Energy": 3,
        "Appetite or Weight Changes": 3,
        "Low Self-Worth": 3,
        "Concentration Difficulties": 3,
        "Psychomotor Changes": 3
    }
}

items_real = ['PHQ8_NoInterest', 'PHQ8_Depressed', 'PHQ8_Sleep', 'PHQ8_Tired', 'PHQ8_Appetite', 'PHQ8_Failure', 'PHQ8_Concentrating', 'PHQ8_Moving']
items_pred = [f'item{i}' for i in range(1, 9)]
item_names = list(item_scoring["PHQ-8"].keys())
vertical_results = []

for i, (item_real, item_pred) in enumerate(zip(items_real, items_pred)):
    real_scores = df_real[item_real]
    pred_scores = df_pred[item_pred]

    if real_scores.sum() == 0 and pred_scores.sum() == 0:
        continue
    num_classes = item_scoring["PHQ-8"][item_names[i]]
    accuracy = accuracy_score(real_scores, pred_scores)

    try:
        macro_f1 = f1_score(real_scores, pred_scores, average='macro')
    except ValueError as e:
        print(f"Macro_F1 calculation error for {item_names[i]}: {str(e)}")
        macro_f1 = np.nan

    vertical_results.append({
        'Item': item_names[i],
        'MAE': mean_absolute_error(real_scores, pred_scores),
        'Accuracy': accuracy,
        'Macro_F1': macro_f1
    })


total_df = pd.DataFrame([total_metrics])
category_df = pd.DataFrame([category_metrics])
items_df = pd.DataFrame(vertical_results)

print("\n=== Total Score Evaluation ===")
print(total_df.round(3))

print("\n=== Depression Classification Evaluation ===")
print(category_df.round(3))

print("\n=== Item-level Evaluation ===")
print(items_df.round(3))