import json
from pathlib import Path

NB = Path('Assignment_Evidence_CBR_with_tuning.ipynb')

nb = json.loads(NB.read_text())
cells = nb.get('cells', [])

def idx_by_id(cell_id: str):
    for i,c in enumerate(cells):
        if c.get('id') == cell_id:
            return i
    return None

move_ids = [
    'ds-hybrid-intro',
    'ds-hybrid-code',
    'ds-soft-top2-intro',
    'ds-soft-top2-code',
]

# Collect cells to move (preserve order)
to_move = []
for cid in move_ids:
    i = idx_by_id(cid)
    if i is not None:
        to_move.append(cells[i])
    else:
        print(f'WARN: cell id {cid} not found')

# Remove them from original positions (remove from end to start to keep indices stable)
for cid in reversed(move_ids):
    i = idx_by_id(cid)
    if i is not None:
        cells.pop(i)

# Find insertion point: after ds-param-sweep-code (or after ds-softvoting-quantile-code if present)
after_ids = ['ds-param-sweep-code', 'ds-softvoting-quantile-code', 'ds-softvoting-code']
ins_idx = None
for aid in after_ids:
    i = idx_by_id(aid)
    if i is not None:
        ins_idx = i + 1
        break
if ins_idx is None:
    # Fallback: append to end
    ins_idx = len(cells)

# Insert moved cells at insertion index
cells[ins_idx:ins_idx] = to_move

nb['cells'] = cells
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print('Reordered cells inserted after index', ins_idx-1)
