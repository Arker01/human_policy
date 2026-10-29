import pathlib,json,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
expected=json.loads((ROOT/'configs/original_source_sha256.json').read_text())
actual={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in expected}
changed=[p for p,h in expected.items() if actual[p]!=h]
checkpoints={}
assert not changed,changed
for name in ('original','geometry'):
 m=json.loads((ROOT/'labels'/name/'manifest.json').read_text());h=hashlib.sha256()
 with open(m['teacher'],'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 assert h.hexdigest()==m['teacher_sha256'],name
 checkpoints[name]=dict(path=m['teacher'],sha256=h.hexdigest(),unchanged=True)
(ROOT/'results/original_integrity.json').write_text(json.dumps(dict(unchanged=True,source_files=len(expected),source_sha256=actual,checkpoints=checkpoints),indent=2))
print('Original source SHA-256 verified:',len(expected),'files; both teacher checkpoints unchanged',flush=True)
