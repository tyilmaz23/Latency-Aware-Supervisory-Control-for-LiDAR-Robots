import csv, gzip, hashlib, importlib.metadata, json, os, platform, sys, tempfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
VERSION="1.0.0"
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def file_hash(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    with open(tmp,"w",encoding="utf-8") as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
    os.replace(tmp,path)
def read_json(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def versions():
    names=["numpy","scipy","scikit-learn","torch","stable-baselines3","gymnasium","requests","matplotlib","websocket-client"]
    result={"python":sys.version,"platform":platform.platform(),"suite":VERSION}
    for name in names:
        try:result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result[name]="missing"
    return result
def input_hashes():
    paths=list((ROOT/"assets").rglob("*"))+list((ROOT/"revision_suite").glob("*.py"))+[ROOT/"baseline_eval_suite.py",ROOT/"trainrl.py"]
    return {str(p.relative_to(ROOT)).replace("\\","/"):file_hash(p) for p in sorted(paths) if p.is_file()}
def rows(path):
    with open(path,encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def csv_write(path,records):
    if not records:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    with open(tmp,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    os.replace(tmp,path)
def scene(env):
    return {"agent":np.asarray(env.agent_pos).astype(float).tolist(),"goal":np.asarray(env.target_pos).astype(float).tolist(),
            "walls":[list(map(float,x)) for x in env.walls],"obstacles":[list(map(float,x)) for x in env.dynamic_walls],
            "moving":[dict(x) for x in env._moving_obstacles]}
def finite_float(x):
    x=float(x)
    return x if np.isfinite(x) else None
