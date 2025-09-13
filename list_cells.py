import json

with open('Assignment_Evidence_CBR.ipynb','r') as f:
    nb=json.load(f)

for i,c in enumerate(nb.get('cells',[]), start=1):
    cid=c.get('id','')
    src=''.join(c.get('source',[]))
    first=(src.splitlines()[0] if src else '').strip()
    print(f"{i:03d} | {c.get('cell_type')} | id={cid} | {first[:80]}")
