import json

with open('Assignment_Evidence_CBR.ipynb','r') as f:
    nb=json.load(f)

for i,c in enumerate(nb.get('cells',[]), start=1):
    outs=c.get('outputs',[])
    if not outs:
        continue
    for o in outs:
        ot=o.get('output_type')
        if ot=='stream':
            txt=o.get('text','')
            if isinstance(txt, list):
                txt=''.join(txt)
            print(f"CELL {i} STREAM: {str(txt).strip()[:500]}")
        d=o.get('data',{})
        if not d:
            continue
        if 'text/plain' in d:
            s=''.join(d['text/plain'])
            print(f"CELL {i} TEXT/PLAIN: {s[:500].replace('\n',' ')}")
        if 'text/html' in d:
            h=''.join(d['text/html'])
            print(f"CELL {i} HTML length: {len(h)}")
        if 'image/png' in d:
            print(f"CELL {i} IMG bytes: {len(d['image/png'])}")
