import json
from pathlib import Path

NB_PATH = Path('Assignment_Evidence_CBR_with_tuning.ipynb')

nb = json.loads(NB_PATH.read_text())
cells = nb.get('cells', [])

def find_index_by_id(cid: str):
    for i, c in enumerate(cells):
        if c.get('id') == cid:
            return i
    return None

def ensure_optimizer_cells():
    md_id = 'ds-optimizer-intro'
    code_id = 'ds-optimizer-code'
    heat_id = 'ds-optimizer-heatmap'

    intro_exists = find_index_by_id(md_id) is not None
    code_exists = find_index_by_id(code_id) is not None
    heat_exists = find_index_by_id(heat_id) is not None

    if intro_exists and code_exists and heat_exists:
        return

    # Build sources
    md_source = [
        "### Optimierungs‑Modus und feineres tau/Ω‑Raster\n",
        "Umschaltbarer Optimierer mit Zielmetrik `OPTIMIZE ∈ {\"accuracy\", \"macro_f1\"}` und feinerem Raster für `tau_singleton` und `Ω_max`. Ergebnis wird erneut nach `Pred_DS_final` geschrieben."
    ]

    code_source = [
        "import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt\n",
        "\n",
        "# Optimierungsziel: 'accuracy' oder 'macro_f1'\n",
        "OPTIMIZE = 'macro_f1'  # 'accuracy' oder 'macro_f1'\n",
        "tau_list = np.arange(0.46, 0.53, 0.01)\n",
        "omega_list = np.arange(0.46, 0.53, 0.01)\n",
        "label_map = {'r':'race bike','m':'mtb','t':'trecking bike'}\n",
        "classes = ['r','m','t']\n",
        "\n",
        "# Prüfen, dass masses und fallback existieren\n",
        "if 'masses' not in globals():\n",
        "    raise RuntimeError('Bitte zunächst die Hybrid‑Zelle ausführen, damit \"masses\" existiert.')\n",
        "fb = None\n",
        "for c in ['Pred_DS_soft_tuned','Pred_DS_soft_best','Pred_DS_soft']:\n",
        "    if c in data.columns: fb = c; break\n",
        "if fb is None:\n",
        "    raise RuntimeError('Bitte Soft‑Voting zuerst ausführen (Pred_DS_soft_tuned oder Pred_DS_soft).')\n",
        "\n",
        "def evaluate_hybrid(tau_singleton, omega_max):\n",
        "    preds = []\n",
        "    for i, row in data.iterrows():\n",
        "        m = masses[i] if i < len(masses) else None\n",
        "        if m is None:\n",
        "            preds.append(row.get(fb, None)); continue\n",
        "        singles = {c: (m[c] if c in m else 0.0) for c in classes}\n",
        "        best_c, best_v = max(singles.items(), key=lambda kv: kv[1])\n",
        "        om_v = float(m[omega]) if omega in m else 0.0\n",
        "        if (best_v >= tau_singleton) and (om_v <= omega_max):\n",
        "            preds.append(label_map[best_c])\n",
        "        else:\n",
        "            preds.append(row.get(fb, None))\n",
        "    df_eval = data[data['Bike Type'].notna() & (data['Bike Type']!='')].copy()\n",
        "    y_true = df_eval['Bike Type']\n",
        "    y_pred = pd.Series(preds, index=data.index).loc[df_eval.index]\n",
        "    # Metriken\n",
        "    cm = pd.crosstab(y_true, y_pred).reindex(index=['race bike','mtb','trecking bike'], columns=['race bike','mtb','trecking bike'], fill_value=0)\n",
        "    diag = np.diag(cm.values); support = cm.sum(axis=1).values; pred_sum = cm.sum(axis=0).values\n",
        "    rec = np.divide(diag, support, out=np.zeros_like(diag, dtype=float), where=support>0)\n",
        "    prec = np.divide(diag, pred_sum, out=np.zeros_like(diag, dtype=float), where=pred_sum>0)\n",
        "    f1 = np.divide(2*prec*rec, prec+rec, out=np.zeros_like(diag, dtype=float), where=(prec+rec)>0)\n",
        "    macro_f1 = float(np.nanmean(f1))\n",
        "    acc = float((y_pred == y_true).mean())\n",
        "    return acc, macro_f1, preds\n",
        "\n",
        "records = []\n",
        "best = (-1.0, -1.0, None, None)\n",
        "best_preds = None\n",
        "for t in tau_list:\n",
        "    for o in omega_list:\n",
        "        acc, mf1, preds = evaluate_hybrid(t, o)\n",
        "        records.append({'tau':t,'omega':o,'acc':acc,'macro_f1':mf1})\n",
        "        if OPTIMIZE=='macro_f1':\n",
        "            better = (mf1 > best[1]) or (mf1 == best[1] and acc > best[0])\n",
        "        else:\n",
        "            better = (acc > best[0]) or (acc == best[0] and mf1 > best[1])\n",
        "        if better:\n",
        "            best = (acc, mf1, t, o); best_preds = preds\n",
        "\n",
        "df_res = pd.DataFrame.from_records(records)\n",
        "acc, mf1, t, o = best\n",
        "print(f'Best Optimizer -> acc={acc:.3f} | macroF1={mf1:.3f} | tau={t} | Ω_max={o} | OPT={OPTIMIZE}')\n",
        "data['Pred_DS_final'] = best_preds\n",
        "# Heatmap\n",
        "pivot = df_res.pivot(index='tau', columns='omega', values=('macro_f1' if OPTIMIZE=='macro_f1' else 'acc'))\n",
        "plt.figure(figsize=(6,4)); sns.heatmap(pivot.sort_index(), annot=True, fmt='.3f', cmap='viridis');\n",
        "plt.title(f'Hybrid Grid {OPTIMIZE}'); plt.tight_layout(); plt.show()\n",
    ]

    heat_source = [
        "# Optional: separate Heatmap (falls in der Optimierer‑Zelle unterdrückt werden soll)\n",
        "import seaborn as sns, matplotlib.pyplot as plt\n",
        "metric = 'macro_f1'  # oder 'acc'\n",
        "pivot = df_res.pivot(index='tau', columns='omega', values=metric)\n",
        "plt.figure(figsize=(6,4)); sns.heatmap(pivot.sort_index(), annot=True, fmt='.3f', cmap='viridis');\n",
        "plt.title(f'Hybrid Grid {metric}'); plt.tight_layout(); plt.show()\n",
    ]

    md_cell = {"cell_type":"markdown","id":md_id,"metadata":{},"source":md_source}
    code_cell = {"cell_type":"code","execution_count":None,"id":code_id,"metadata":{},"outputs":[],"source":code_source}
    heat_cell = {"cell_type":"code","execution_count":None,"id":heat_id,"metadata":{},"outputs":[],"source":heat_source}

    # Insert after ds-hybrid-grid-code, else append at end
    anchor = find_index_by_id('ds-hybrid-grid-code')
    ins_at = anchor + 1 if anchor is not None else len(cells)
    cells[ins_at:ins_at] = [md_cell, code_cell, heat_cell]

ensure_optimizer_cells()
nb['cells'] = cells
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print('Inserted optimizer cells.')

