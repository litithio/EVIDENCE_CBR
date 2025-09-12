import pandas as pd
import numpy as np

# Read CSV
df = pd.read_csv('bikedata/sampled_data_001.csv', sep=';', decimal=',')

def analyze_feature(series):
    s = series.astype(float)
    mean_v = np.nanmean(s); min_v = np.nanmin(s); max_v = np.nanmax(s)
    tol = (max_v - min_v) * 0.10
    def low(x):
        if x <= mean_v:
            return (mean_v - x) / (mean_v - min_v) if mean_v > min_v else 0.0
        return 0.0
    def high(x):
        if x >= mean_v:
            return (x - mean_v) / (max_v - mean_v) if max_v > mean_v else 0.0
        return 0.0
    def medium(x):
        if (mean_v - tol) <= x <= mean_v:
            return (x - (mean_v - tol)) / tol if tol > 0 else 0.0
        elif mean_v < x <= (mean_v + tol):
            return ((mean_v + tol) - x) / tol if tol > 0 else 0.0
        return 0.0
    return low, medium, high

dist_low, dist_med, dist_high = analyze_feature(df['Distance'])
elev_low, elev_med, elev_high = analyze_feature(df['Elevation Gain'])

CLASSES = ['race bike','mtb','trecking bike']
map_dist = {'low':'mtb','medium':'trecking bike','high':'race bike'}
map_elev = {'low':'trecking bike','medium':'race bike','high':'mtb'}

def best_category(funcs, x):
    vals = {name: max(0.0, min(1.0, f(x))) for name, f in funcs.items()}
    cat = max(vals, key=vals.get)
    return cat, vals[cat]

def mass_from(cat, mu, alpha=0.8):
    # singleton mass for cat, rest to omega
    m = {c: 0.0 for c in CLASSES}
    m[cat] = alpha * mu
    m['omega'] = 1 - m[cat]
    return m

def combine_two(m1, m2):
    # specialized DS combine for singletons + omega
    classes = CLASSES
    k_conf = 0.0
    for i in classes:
        for j in classes:
            if i != j:
                k_conf += m1.get(i,0.0)*m2.get(j,0.0)
    one_minus_k = 1.0 - k_conf
    if one_minus_k <= 1e-12:
        # fallback to average
        avg = {c: 0.5*(m1.get(c,0.0)+m2.get(c,0.0)) for c in classes}
        avg['omega'] = min(1.0, max(0.0, 0.5*(m1.get('omega',0.0)+m2.get('omega',0.0))))
        return avg
    comb = {}
    for c in classes:
        comb[c] = (m1.get(c,0.0)*m2.get(c,0.0) + m1.get(c,0.0)*m2.get('omega',0.0) + m1.get('omega',0.0)*m2.get(c,0.0)) / one_minus_k
    comb['omega'] = (m1.get('omega',0.0)*m2.get('omega',0.0)) / one_minus_k
    return comb

alpha=0.8
preds=[]
omegas=[]
for _,row in df.iterrows():
    try:
        d=float(row['Distance']); e=float(row['Elevation Gain'])
    except Exception:
        preds.append(None); omegas.append(np.nan); continue
    cd, mu_d = best_category({'low':dist_low,'medium':dist_med,'high':dist_high}, d)
    ce, mu_e = best_category({'low':elev_low,'medium':elev_med,'high':elev_high}, e)
    m1=mass_from(map_dist[cd], mu_d, alpha)
    m2=mass_from(map_elev[ce], mu_e, alpha)
    m3=combine_two(m1,m2)
    omegas.append(m3['omega'])
    best = max(CLASSES, key=lambda c: m3.get(c,0.0))
    preds.append(best)

df['Pred_DS'] = preds
df['Omega'] = omegas

df_eval = df[df['Bike Type'].notna() & (df['Bike Type']!='')].copy()
acc = (df_eval['Pred_DS']==df_eval['Bike Type']).mean()
print(f"Accuracy: {acc:.3f}  n={len(df_eval)}  alpha={alpha}")

cm = pd.crosstab(df_eval['Bike Type'], df_eval['Pred_DS']).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
print('\nConfusion matrix (counts):')
print(cm)
cmn = cm.div(cm.sum(axis=1), axis=0)
print('\nConfusion matrix (row-normalized):')
print(cmn.round(3))

diag = np.diag(cm.values)
support = cm.sum(axis=1).values
pred_sum = cm.sum(axis=0).values
recall = np.divide(diag, support, out=np.zeros_like(diag, dtype=float), where=support>0)
precision = np.divide(diag, pred_sum, out=np.zeros_like(diag, dtype=float), where=pred_sum>0)
f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros_like(diag, dtype=float), where=(precision+recall)>0)

print('\nPer-class metrics:')
for cls, p,r,f,s in zip(CLASSES, precision, recall, f1, support):
    print(f"{cls:14s}  P={p:.3f}  R={r:.3f}  F1={f:.3f}  support={int(s)}")

# Omega quartile analysis
df_eval_q = df_eval.copy()
df_eval_q['Correct'] = (df_eval_q['Pred_DS']==df_eval_q['Bike Type'])
df_eval_q = df_eval_q.dropna(subset=['Omega'])
if len(df_eval_q)>0:
    df_eval_q['OmegaBin'] = pd.qcut(df_eval_q['Omega'], q=4, duplicates='drop')
    acc_by_bin = df_eval_q.groupby('OmegaBin')['Correct'].mean()
    print('\nAccuracy by Omega quartiles:')
    for b,v in acc_by_bin.items():
        print(f"{b}: {v:.3f}")
    print('\nOmega mean/std:', df_eval_q['Omega'].mean().round(3), df_eval_q['Omega'].std().round(3))
