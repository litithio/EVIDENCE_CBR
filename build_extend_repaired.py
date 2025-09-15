import json
from pathlib import Path

NBP = Path('Assignment_Evidence_CBR_with_tuning_REPAIRED.ipynb')
nb = json.loads(NBP.read_text())
cells = nb.get('cells', [])

def md(text):
    return {"cell_type":"markdown","metadata":{},"source":[s+"\n" for s in text.splitlines()]}

def code(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[s+"\n" for s in src.splitlines()]}

# Helper to prevent duplicates by checking first line markers
def has_cell_with_first_line(prefix:str)->bool:
    for c in cells:
        if c.get('cell_type')=='code':
            src=c.get('source',[])
            if src and src[0].lstrip().startswith(prefix):
                return True
    return False

# 1) Baseline Snapshot (if not present)
if not has_cell_with_first_line('import numpy as np, pandas as pd  # baseline snapshot'):
    cells.append(md('### Baseline‑Metriken festhalten'))
    cells.append(code("""
import numpy as np, pandas as pd  # baseline snapshot
df_eval_base = data[data['Bike Type'].notna() & (data['Bike Type']!='')].copy()
acc_base = float((df_eval_base['Pred_DS'] == df_eval_base['Bike Type']).mean())
cm_base = pd.crosstab(df_eval_base['Bike Type'], df_eval_base['Pred_DS']).reindex(index=['race bike','mtb','trecking bike'], columns=['race bike','mtb','trecking bike'], fill_value=0)
diag = np.diag(cm_base.values); support = cm_base.sum(axis=1).values; pred_sum = cm_base.sum(axis=0).values
rec_base = np.divide(diag, support, out=np.zeros_like(diag, float), where=support>0)
prec_base = np.divide(diag, pred_sum, out=np.zeros_like(diag, float), where=pred_sum>0)
f1_base = np.divide(2*prec_base*rec_base, prec_base+rec_base, out=np.zeros_like(diag, float), where=(prec_base+rec_base)>0)
baseline_metrics = {
    'accuracy': acc_base,
    'per_class': {cls: {'precision': float(p), 'recall': float(r), 'f1': float(f), 'support': int(s)} for cls,p,r,f,s in zip(cm_base.index, prec_base, rec_base, f1_base, support)}
}
print('Baseline Accuracy:', round(acc_base,3))
for cls,st in baseline_metrics['per_class'].items():
    print(f"  {cls:14s} P={st['precision']:.3f} R={st['recall']:.3f} F1={st['f1']:.3f} (n={st['support']})")
"""))

# 2) Ω‑Analyse und alpha‑Sensitivität (optional diagnostics)
if not has_cell_with_first_line('# Omega/Alpha Diagnose'):
    cells.append(md('### Diagnose: Ω‑Verteilung und Alpha‑Sensitivität'))
    cells.append(code("""
# Omega/Alpha Diagnose
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
if 'masses' in globals():
    omega_vals = [ (float(m[omega]) if m is not None and (omega in m) else np.nan) for m in masses ]
    data['OmegaMass'] = omega_vals
    df_eval = data[data['Bike Type'].notna() & (data['Bike Type']!='')].copy()
    df_eval = df_eval.dropna(subset=['OmegaMass'])
    if len(df_eval)>0:
        df_eval['OmegaBin'] = pd.qcut(df_eval['OmegaMass'], q=4, duplicates='drop')
        acc_by = df_eval.assign(Correct=(df_eval['Pred_DS']==df_eval['Bike Type'])).groupby('OmegaBin')['Correct'].mean()
        plt.figure(figsize=(6,4)); acc_by.plot(kind='bar'); plt.ylabel('Accuracy'); plt.title('Accuracy nach Ω‑Quartilen'); plt.tight_layout(); plt.show()
        plt.figure(figsize=(6,4)); sns.histplot(df_eval['OmegaMass'], bins=20, kde=True); plt.title('Verteilung Ω'); plt.tight_layout(); plt.show()
else:
    print('masses nicht vorhanden – bitte Hybrid/DS Abschnitt vorher ausführen.')

def eval_alpha(alpha_val: float) -> float:
    preds_tmp=[]
    for _,row in data.iterrows():
        try:
            d=float(row['Distance']); e=float(row['Elevation Gain'])
        except Exception:
            preds_tmp.append(None); continue
        # Winner‑takes‑all auf Baseline‑Fuzzy
        cat_d, mu_d = max({'low': dist_low(d), 'medium': dist_med(d), 'high': dist_high(d)}.items(), key=lambda kv: kv[1])
        cat_e, mu_e = max({'low': elev_low(e), 'medium': elev_med(e), 'high': elev_high(e)}.items(), key=lambda kv: kv[1])
        m1 = MassFunction({{'low': 'm', 'medium': 't', 'high': 'r'}[cat_d]: alpha_val*mu_d, omega: 1-alpha_val*mu_d})
        m2 = MassFunction({{'low': 't', 'medium': 'r', 'high': 'm'}[cat_e]: alpha_val*mu_e, omega: 1-alpha_val*mu_e})
        m3 = m1 & m2
        best = max(['r','m','t'], key=lambda c: (m3[c] if c in m3 else 0.0))
        preds_tmp.append({'r':'race bike','m':'mtb','t':'trecking bike'}[best])
    df_eval = data[data['Bike Type'].notna() & (data['Bike Type']!='')]
    return float((pd.Series(preds_tmp, index=data.index).loc[df_eval.index]==df_eval['Bike Type']).mean())

alphas = np.linspace(0.5,0.95,10)
accs = [eval_alpha(a) for a in alphas]
plt.figure(figsize=(6,4)); plt.plot(alphas, accs, marker='o'); plt.grid(True); plt.xlabel('alpha'); plt.ylabel('Accuracy'); plt.title('Alpha‑Sensitivität (Baseline‑WTA)'); plt.tight_layout(); plt.show()
"""))

# 3) Final‑Vergleichsvisuals (falls noch nicht vorhanden)
def has_title(title:str)->bool:
    for c in cells:
        if c.get('cell_type')=='markdown':
            s=''.join(c.get('source',[]))
            if s.strip().startswith(title):
                return True
    return False

if not has_title('### Visual: Baseline vs Final'):
    cells.append(md('### Visual: Baseline vs Final (Precision/Recall/F1)'))
    cells.append(code("""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
df_eval = data[data['Bike Type'].notna() & (data['Bike Type']!='')].copy()
classes = ['race bike','mtb','trecking bike']
if 'baseline_metrics' in globals():
    base_prec=[baseline_metrics['per_class'][c]['precision'] for c in classes]
    base_rec =[baseline_metrics['per_class'][c]['recall'] for c in classes]
    base_f1  =[baseline_metrics['per_class'][c]['f1'] for c in classes]
else:
    cm_b = pd.crosstab(df_eval['Bike Type'], df_eval['Pred_DS']).reindex(index=classes, columns=classes, fill_value=0)
    diag_b=np.diag(cm_b.values); support_b=cm_b.sum(axis=1).values; pred_sum_b=cm_b.sum(axis=0).values
    base_rec=np.divide(diag_b,support_b,out=np.zeros_like(diag_b,float),where=support_b>0)
    base_prec=np.divide(diag_b,pred_sum_b,out=np.zeros_like(diag_b,float),where=pred_sum_b>0)
    base_f1=np.divide(2*base_prec*base_rec, base_prec+base_rec, out=np.zeros_like(diag_b,float), where=(base_prec+base_rec)>0)
cm_f = pd.crosstab(df_eval['Bike Type'], df_eval['Pred_DS_final']).reindex(index=classes, columns=classes, fill_value=0)
diag_f=np.diag(cm_f.values); support_f=cm_f.sum(axis=1).values; pred_sum_f=cm_f.sum(axis=0).values
fin_rec=np.divide(diag_f,support_f,out=np.zeros_like(diag_f,float),where=support_f>0)
fin_prec=np.divide(diag_f,pred_sum_f,out=np.zeros_like(diag_f,float),where=pred_sum_f>0)
fin_f1=np.divide(2*fin_prec*fin_rec, fin_prec+fin_rec, out=np.zeros_like(diag_f,float), where=(fin_prec+fin_rec)>0)
fig,axes=plt.subplots(1,3, figsize=(12,4), sharey=False)
metrics=[('Precision', base_prec, fin_prec), ('Recall', base_rec, fin_rec), ('F1', base_f1, fin_f1)]
x=np.arange(len(classes)); width=0.38
for ax,(title,base_vals,fin_vals) in zip(axes,metrics):
    ax.bar(x-width/2, base_vals, width, label='Baseline')
    ax.bar(x+width/2, fin_vals,  width, label='Final')
    ax.set_title(title); ax.set_xticks(x); ax.set_xticklabels(classes, rotation=15)
    ax.set_ylim(0,1); ax.grid(True, axis='y', alpha=0.3)
axes[0].set_ylabel('Wert'); axes[-1].legend(loc='lower right'); plt.tight_layout(); plt.show()
"""))

if not has_title('### Visual: Scatter Final richtig/falsch'):
    cells.append(md('### Visual: Scatter Final richtig/falsch'))
    cells.append(code("""
import matplotlib.pyplot as plt
df_eval = data[data['Bike Type'].notna() & (data['Bike Type']!='')].copy()
df_eval['CorrectFinal'] = (df_eval['Pred_DS_final']==df_eval['Bike Type'])
markers={'race bike':'o','mtb':'s','trecking bike':'^'}
plt.figure(figsize=(7,5))
for truth, sub in df_eval.groupby('Bike Type'):
    plt.scatter(sub['Distance'], sub['Elevation Gain'], c=sub['CorrectFinal'].map({True:'tab:green',False:'tab:red'}),
                marker=markers.get(truth,'o'), alpha=0.8, edgecolor='k', linewidths=0.2, label=truth)
plt.xlabel('Distance'); plt.ylabel('Elevation Gain'); plt.title('Final: richtig (grün) vs. falsch (rot)'); plt.grid(True, alpha=0.3); plt.legend(title='Wahre Klasse'); plt.tight_layout(); plt.show()
"""))

nb['cells']=cells
NBP.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print('Extended REPAIRED notebook.')
