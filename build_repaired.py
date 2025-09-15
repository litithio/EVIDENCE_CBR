import json
from pathlib import Path

BASE = Path('Assignment_Evidence_CBR.ipynb')
OUT = Path('Assignment_Evidence_CBR_with_tuning_REPAIRED.ipynb')

nb = json.loads(BASE.read_text())
cells = list(nb.get('cells', []))

def md(text):
    return {"cell_type":"markdown","metadata":{},"source":[s+"\n" for s in text.splitlines()]}

def code(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[s+"\n" for s in src.splitlines()]}

# Extension cells appended in logical order

cells.append(md("""
### Erweiterte Bewertung (Evidenztheorie)

Wir ergänzen Konfusionsmatrix und per‑Klasse‑Metriken.
"""))

cells.append(code("""
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
df_eval = data[data['Bike Type'].notna() & (data['Bike Type']!='')].copy()
cm = pd.crosstab(df_eval['Bike Type'], df_eval['Pred_DS']).reindex(index=['race bike','mtb','trecking bike'], columns=['race bike','mtb','trecking bike'], fill_value=0)
cmn = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
plt.figure(figsize=(6,4)); sns.heatmap(cmn, annot=True, fmt='.2f', cmap='Blues'); plt.title('Konfusionsmatrix (zeilennormiert)'); plt.tight_layout(); plt.show()
diag = np.diag(cm.values); support = cm.sum(axis=1).values; pred_sum = cm.sum(axis=0).values
rec = np.divide(diag, support, out=np.zeros_like(diag, float), where=support>0)
prec = np.divide(diag, pred_sum, out=np.zeros_like(diag, float), where=pred_sum>0)
f1 = np.divide(2*prec*rec, prec+rec, out=np.zeros_like(diag, float), where=(prec+rec)>0)
report = pd.DataFrame({'precision':prec,'recall':rec,'f1':f1,'support':support}, index=cm.index)
display(report.round(3))
"""))

cells.append(md("""
### DS mit Soft‑Voting (anstatt Winner‑takes‑all)
"""))

cells.append(code("""
alpha_dist = 0.8
alpha_elev = 0.7
label_map = {'r':'race bike','m':'mtb','t':'trecking bike'}
CLASSES = ['r','m','t']

def mass_from_mu(mu_by_cls, alpha):
    s = sum(mu_by_cls.values())
    if s <= 0: return MassFunction({omega:1.0})
    m = {c: alpha*(mu_by_cls.get(c,0.0)/s) for c in CLASSES}; m[omega] = max(0.0, 1.0-sum(m.values()))
    return MassFunction(m)

pred_soft=[]
for _,row in data.iterrows():
    try:
        d=float(row['Distance']); e=float(row['Elevation Gain'])
    except Exception:
        pred_soft.append(None); continue
    mu_d={'r':dist_high(d),'m':dist_low(d),'t':dist_med(d)}
    mu_e={'m':elev_high(e),'r':elev_med(e),'t':elev_low(e)}
    m1=mass_from_mu(mu_d, alpha_dist); m2=mass_from_mu(mu_e, alpha_elev); m=m1 & m2
    best=max(CLASSES, key=lambda c: (m[c] if c in m else 0.0))
    pred_soft.append(label_map[best])
data['Pred_DS_soft']=pred_soft

df_eval = data[data['Bike Type'].notna() & (data['Bike Type']!='')]
print('Accuracy (DS soft):', float((df_eval['Pred_DS_soft']==df_eval['Bike Type']).mean()))
"""))

cells.append(md("""
### Optional: Sanfte Quantile + Soft‑Voting
"""))

cells.append(code("""
def trimf(a,b,c):
    def f(x):
        if x<=a or x>=c: return 0.0
        if x==b: return 1.0
        if x<b: return (x-a)/(b-a) if b>a else 0.0
        return (c-x)/(c-b) if c>b else 0.0
    return f

qD = data['Distance'].quantile([0.0,0.35,0.5,0.65,1.0]).to_dict()
D_min,D_q35,D_q50,D_q65,D_max = qD[0.0],qD[0.35],qD[0.5],qD[0.65],qD[1.0]
dist_low_s=trimf(D_min,D_q35,D_q50); dist_med_s=trimf(D_q35,D_q50,D_q65); dist_high_s=trimf(D_q50,D_q65,D_max)

qE = data['Elevation Gain'].quantile([0.0,0.50,0.70,0.85,1.0]).to_dict()
E_min,E_q50,E_q70,E_q85,E_max = qE[0.0],qE[0.50],qE[0.70],qE[0.85],qE[1.0]
elev_low_s=trimf(E_min,E_q50,E_q70); elev_med_s=trimf(E_q50,(E_q50+E_q70)/2.0,E_q85); elev_high_s=trimf(E_q70,E_q85,E_max)

pred_soft_tuned=[]
for _,row in data.iterrows():
    try:
        d=float(row['Distance']); e=float(row['Elevation Gain'])
    except Exception:
        pred_soft_tuned.append(None); continue
    mu_d={'r':dist_high_s(d),'m':dist_low_s(d),'t':dist_med_s(d)}
    mu_e={'m':elev_high_s(e),'r':elev_med_s(e),'t':elev_low_s(e)}
    m1=mass_from_mu(mu_d, alpha_dist); m2=mass_from_mu(mu_e, alpha_elev); m=m1 & m2
    best=max(CLASSES, key=lambda c: (m[c] if c in m else 0.0))
    pred_soft_tuned.append(label_map[best])
data['Pred_DS_soft_tuned']=pred_soft_tuned
"""))

cells.append(md("""
### Hybrid: DS (Baseline) bei hoher Sicherheit, sonst Soft‑Voting
"""))

cells.append(code("""
tau_singleton=0.48; omega_max=0.48
label_map={'r':'race bike','m':'mtb','t':'trecking bike'}
fallback_cols=[c for c in ['Pred_DS_soft_tuned','Pred_DS_soft','Pred_DS_soft_best'] if c in data.columns]
fallback_col=fallback_cols[0] if fallback_cols else None
if fallback_col is None:
    raise RuntimeError('Kein Soft‑Voting Ergebnis gefunden.')
preds_hybrid=[]
for i,row in data.iterrows():
    m=masses[i] if i<len(masses) else None
    if m is None: preds_hybrid.append(row.get(fallback_col,None)); continue
    singles={c:(m[c] if c in m else 0.0) for c in ['r','m','t']}
    best_c,best_v=max(singles.items(), key=lambda kv: kv[1])
    om_v=float(m[omega]) if omega in m else 0.0
    preds_hybrid.append(label_map[best_c] if (best_v>=tau_singleton and om_v<=omega_max) else row.get(fallback_col,None))
data['Pred_DS_hybrid']=preds_hybrid
df_eval=data[data['Bike Type'].notna() & (data['Bike Type']!='')].copy()
acc=float((df_eval['Pred_DS_hybrid']==df_eval['Bike Type']).mean())
print(f'Accuracy (DS hybrid): {acc:.3f} | tau={tau_singleton}, Ω_max={omega_max}')
"""))

cells.append(md("""
### Optimizer – Hybrid tau/Ω mit OPTIMIZE‑Schalter
"""))

cells.append(code("""
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
OPTIMIZE='macro_f1'  # 'accuracy' | 'macro_f1'
tau_list=np.arange(0.46,0.53,0.01); omega_list=np.arange(0.46,0.53,0.01)
label_map={'r':'race bike','m':'mtb','t':'trecking bike'}; classes=['r','m','t']
if 'masses' not in globals(): raise RuntimeError('Bitte Hybrid zuerst ausführen (masses fehlt).')
fb=None
for c in ['Pred_DS_soft_tuned','Pred_DS_soft_best','Pred_DS_soft']:
    if c in data.columns: fb=c; break
if fb is None: raise RuntimeError('Bitte Soft‑Voting zuerst ausführen (Pred_DS_soft_tuned oder Pred_DS_soft).')
def evaluate_hybrid(tau_singleton, omega_max):
    preds=[]
    for i,row in data.iterrows():
        m=masses[i] if i<len(masses) else None
        if m is None: preds.append(row.get(fb,None)); continue
        singles={c:(m[c] if c in m else 0.0) for c in classes}
        best_c,best_v=max(singles.items(), key=lambda kv: kv[1])
        om_v=float(m[omega]) if omega in m else 0.0
        preds.append(label_map[best_c] if (best_v>=tau_singleton and om_v<=omega_max) else row.get(fb,None))
    df_eval=data[data['Bike Type'].notna() & (data['Bike Type']!='')].copy()
    y_true=df_eval['Bike Type']; y_pred=pd.Series(preds, index=data.index).loc[df_eval.index]
    cm=pd.crosstab(y_true,y_pred).reindex(index=['race bike','mtb','trecking bike'], columns=['race bike','mtb','trecking bike'], fill_value=0)
    diag=np.diag(cm.values); support=cm.sum(axis=1).values; pred_sum=cm.sum(axis=0).values
    rec=np.divide(diag,support,out=np.zeros_like(diag,float),where=support>0)
    prec=np.divide(diag,pred_sum,out=np.zeros_like(diag,float),where=pred_sum>0)
    f1=np.divide(2*prec*rec,prec+rec,out=np.zeros_like(diag,float),where=(prec+rec)>0)
    acc=float((y_pred==y_true).mean()); macro_f1=float(np.nanmean(f1))
    return acc, macro_f1, preds
records=[]; best=(-1.0,-1.0,None,None); best_preds=None
for t in tau_list:
    for o in omega_list:
        acc,mf1,preds=evaluate_hybrid(t,o)
        records.append({'tau':t,'omega':o,'acc':acc,'macro_f1':mf1})
        better=(mf1>best[1] or (mf1==best[1] and acc>best[0])) if OPTIMIZE=='macro_f1' else (acc>best[0] or (acc==best[0] and mf1>best[1]))
        if better: best=(acc,mf1,t,o); best_preds=preds
df_res=pd.DataFrame.from_records(records)
acc,mf1,t,o=best
print(f"Best Optimizer -> acc={acc:.3f} | macroF1={mf1:.3f} | tau={t} | Ω_max={o} | OPT={OPTIMIZE}")
data['Pred_DS_final']=best_preds
metric='macro_f1' if OPTIMIZE=='macro_f1' else 'acc'
pivot=df_res.pivot(index='tau', columns='omega', values=metric)
plt.figure(figsize=(6,4)); sns.heatmap(pivot.sort_index(), annot=True, fmt='.3f', cmap='viridis'); plt.title(f'Hybrid Grid {OPTIMIZE}'); plt.tight_layout(); plt.show()
"""))

cells.append(md("""
### Finale Auswertung (Pred_DS_final)
"""))

cells.append(code("""
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
df_eval=data[data['Bike Type'].notna() & (data['Bike Type']!='')].copy()
cm_f=pd.crosstab(df_eval['Bike Type'], df_eval['Pred_DS_final']).reindex(index=['race bike','mtb','trecking bike'], columns=['race bike','mtb','trecking bike'], fill_value=0)
diag_f=np.diag(cm_f.values); support_f=cm_f.sum(axis=1).values; pred_sum_f=cm_f.sum(axis=0).values
rec_f=np.divide(diag_f,support_f,out=np.zeros_like(diag_f,float),where=support_f>0)
prec_f=np.divide(diag_f,pred_sum_f,out=np.zerosLike(diag_f,float),where=pred_sum_f>0) if False else np.divide(diag_f,pred_sum_f,out=np.zeros_like(diag_f,float),where=pred_sum_f>0)
f1_f=np.divide(2*prec_f*rec_f,prec_f+rec_f,out=np.zeros_like(diag_f,float),where=(prec_f+rec_f)>0)
macro_f1_f=float(np.nanmean(f1_f)); acc_f=float((df_eval['Pred_DS_final']==df_eval['Bike Type']).mean())
plt.figure(figsize=(6,4)); sns.heatmap(cm_f.div(cm_f.sum(axis=1).replace(0,np.nan), axis=0).fillna(0), annot=True, fmt='.2f', cmap='Greens'); plt.title('Konfusionsmatrix Final (zeilennormiert)'); plt.tight_layout(); plt.show()
print('Per‑Klasse (Final) P/R/F1:')
for cls,p,r,f,s in zip(cm_f.index, prec_f, rec_f, f1_f, support_f):
    print(f"{cls:14s}  P={p:.3f}  R={r:.3f}  F1={f:.3f}  (n={int(s)})")
"""))

nb['cells']=cells
nb['metadata']=nb.get('metadata', {})
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print('Wrote', OUT)
