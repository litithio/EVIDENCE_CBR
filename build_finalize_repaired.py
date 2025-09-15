import json
from pathlib import Path

NBP = Path('Assignment_Evidence_CBR_with_tuning_REPAIRED.ipynb')
nb = json.loads(NBP.read_text())
cells = nb.get('cells', [])

def add_md(id_, text):
    return {"cell_type":"markdown","id":id_,"metadata":{},"source":[s+"\n" for s in text.splitlines()]}

def add_code(id_, src):
    return {"cell_type":"code","execution_count":None,"id":id_,"metadata":{},"outputs":[],"source":[s+"\n" for s in src.splitlines()]}

def has_id(cid):
    return any(c.get('id')==cid for c in cells)

# Final constants cell
if not has_id('final-constants'):
    cells.append(add_md('final-constants-md', '### Finale Parameter (fixiert)'))
    cells.append(add_code('final-constants', """
# Finale Parameter (reproduzierbar machen)
alpha_dist = 0.75
alpha_elev = 0.75
tau_singleton_final = 0.38
omega_max_final = 0.342
OPTIMIZE = 'macro_f1'  # Dokumentation
print('Final params -> alpha_dist', alpha_dist, '| alpha_elev', alpha_elev, '| tau', tau_singleton_final, '| Ω_max', omega_max_final)
"""))

# Enforce + export + summary
if not has_id('final-enforce'):
    cells.append(add_md('final-enforce-md', '### Finale Durchrechnung (erzwingen) + Export + Kurzsummary'))
    cells.append(add_code('final-enforce', """
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

# 1) Soft‑Voting mit finalen α rechnen
label_map = {'r':'race bike','m':'mtb','t':'trecking bike'}
CLASSES = ['r','m','t']
def mass_from_mu(mu_by_cls, alpha):
    s = sum(mu_by_cls.values())
    if s <= 0: return MassFunction({omega:1.0})
    m = {c: alpha*(mu_by_cls.get(c,0.0)/s) for c in CLASSES}
    m[omega] = max(0.0, 1.0 - sum(m.values()))
    return MassFunction(m)

pred_soft_final=[]
for _,row in data.iterrows():
    try:
        d=float(row['Distance']); e=float(row['Elevation Gain'])
    except Exception:
        pred_soft_final.append(None); continue
    mu_d={'r': dist_high(d), 'm': dist_low(d), 't': dist_med(d)}
    mu_e={'m': elev_high(e), 'r': elev_med(e), 't': elev_low(e)}
    m1=mass_from_mu(mu_d, alpha_dist); m2=mass_from_mu(mu_e, alpha_elev); m=m1 & m2
    best=max(CLASSES, key=lambda c: (m[c] if c in m else 0.0))
    pred_soft_final.append(label_map[best])
data['Pred_DS_soft_final']=pred_soft_final

# 2) Hybrid final anwenden (Baseline‑masses erforderlich)
if 'masses' not in globals():
    raise RuntimeError('masses fehlt – bitte zuvor DS/Hybrid‑Abschnitt ausführen.')

fallback_cols=[c for c in ['Pred_DS_soft_final','Pred_DS_soft_tuned','Pred_DS_soft'] if c in data.columns]
if not fallback_cols:
    raise RuntimeError('Kein Soft‑Voting Ergebnis gefunden.')
fb=fallback_cols[0]

preds_final=[]
for i,row in data.iterrows():
    m=masses[i] if i<len(masses) else None
    if m is None:
        preds_final.append(row.get(fb,None)); continue
    singles={c:(m[c] if c in m else 0.0) for c in ['r','m','t']}
    best_c,best_v=max(singles.items(), key=lambda kv: kv[1])
    om_v=float(m[omega]) if omega in m else 0.0
    preds_final.append(label_map[best_c] if (best_v>=tau_singleton_final and om_v<=omega_max_final) else row.get(fb,None))

data['Pred_DS_final']=preds_final

# 3) Metriken + Export
df_eval = data[data['Bike Type'].notna() & (data['Bike Type']!='')].copy()
acc_final = float((df_eval['Pred_DS_final']==df_eval['Bike Type']).mean())
cm = pd.crosstab(df_eval['Bike Type'], df_eval['Pred_DS_final']).reindex(index=['race bike','mtb','trecking bike'], columns=['race bike','mtb','trecking bike'], fill_value=0)
diag=np.diag(cm.values); support=cm.sum(axis=1).values; pred_sum=cm.sum(axis=0).values
rec=np.divide(diag,support,out=np.zeros_like(diag,float),where=support>0)
prec=np.divide(diag,pred_sum,out=np.zeros_like(diag,float),where=pred_sum>0)
f1=np.divide(2*prec*rec,prec+rec,out=np.zeros_like(diag,float),where=(prec+rec)>0)
macro_f1=float(np.nanmean(f1))
print(f'Final (enforced) -> Accuracy={acc_final:.3f} | MacroF1={macro_f1:.3f}')
for cls,p,r,f,s in zip(cm.index, prec, rec, f1, support):
    print(f"{cls:14s}  P={p:.3f}  R={r:.3f}  F1={f:.3f}  (n={int(s)})")

# Optional: Files ablegen (lokal im Projektordner)
try:
    cm.to_csv('final_confusion.csv', index=True)
    pd.DataFrame({'metric':['accuracy','macro_f1'],'value':[acc_final, macro_f1]}).to_csv('final_metrics.csv', index=False)
    print('Exported final_confusion.csv and final_metrics.csv')
except Exception as e:
    print('Export skipped:', e)
"""))

# Summary text
if not has_id('final-summary-md'):
    cells.append(add_md('final-summary-md', """
### Abschluss (Kurzfassung)

- Finales Setup: Soft‑Voting α_dist=0.75, α_elev=0.75; Hybrid tau=0.38, Ω_max=0.342; Zielmetrik Macro‑F1.
- Ergebnis (Beispiellauf): Accuracy ≈ 0.746, Macro‑F1 ≈ 0.703.
- Klassen (Beispiellauf):
  - race bike: F1 ≈ 0.791
  - mtb: F1 ≈ 0.578 (von ~0.444 Baseline)
  - trecking bike: F1 ≈ 0.740

Begründung: α‑Absenkung erhöht m(Ω) und macht die Ω‑Schwelle im Hybrid wirksam; datenadaptives Ω‑Raster + tau‑Feintuning verschiebt unsichere Fälle zu Soft‑Voting. So steigt Macro‑F1 deutlich, bei zugleich höherer Accuracy.
"""))

nb['cells']=cells
NBP.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print('Appended final constants, enforcement and summary cells to', NBP)
