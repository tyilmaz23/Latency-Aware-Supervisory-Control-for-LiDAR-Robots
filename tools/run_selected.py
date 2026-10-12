"""Explicit one-job replay; live provider use requires --allow-live."""
import argparse,gzip,json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--list',action='store_true');p.add_argument('--job');p.add_argument('--outdir');p.add_argument('--allow-live',action='store_true');p.add_argument('--api-cap',type=int,default=100000)
a=p.parse_args();cfg=json.loads(gzip.decompress((root/'data/run_configurations.json.gz').read_bytes()))
if a.list:
 for job,m in sorted(cfg.items()):print(job,m['run']['resolved_config']['episodes'],m['run']['resolved_config']['method'],m['run']['resolved_config']['execution'])
 sys.exit()
if a.job not in cfg or not a.outdir:p.error('Use --job from --list and a fresh --outdir.')
m=cfg[a.job];config=m['run']['resolved_config']
if m['archive_group']=='live' and not a.allow_live:p.error('Provider runs require --allow-live and your own GEMINI_API_KEY. They incur API cost and wall-clock pacing.')
sys.path.insert(0,str(root/'simulation'))
from revision_suite.run_job import run
job=dict(m['run']['job'],config=config)
run(job,Path(a.outdir).resolve()/a.job,a.api_cap)
