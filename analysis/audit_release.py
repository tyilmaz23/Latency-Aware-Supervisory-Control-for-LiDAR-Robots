"""Offline audit of selected records; no provider calls or simulations."""
import gzip,hashlib,json,math,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[1]
def read(p):
 p=ROOT/p
 return json.loads(gzip.decompress(p.read_bytes()).decode('utf-8-sig')) if p.suffix=='.gz' else json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):
 with gzip.open(p,'rt',encoding='utf-8-sig') as f:
  for line in f:
   if line.strip():yield json.loads(line)
def wilson(k,n):
    if not n:return [None,None]
    z=1.95996398454;p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [mid-half,mid+half]

def paired(a,b,endpoint="reached",bootstrap=2000):
    """A minus B; strict seed/scene match. Prefix pairing must be declared by caller."""
    aa={int(r["seed"]):r for r in a};bb={int(r["seed"]):r for r in b}
    if set(aa)!=set(bb):raise ValueError("Unequal paired seed sets")
    seeds=sorted(aa)
    if any(aa[s]["scene_hash"]!=bb[s]["scene_hash"] for s in seeds):raise ValueError("Paired scene hash mismatch")
    x=np.array([aa[s]["outcome"]==endpoint for s in seeds],int)
    y=np.array([bb[s]["outcome"]==endpoint for s in seeds],int);d=x-y;n=len(d)
    a_only=int(((x==1)&(y==0)).sum());b_only=int(((x==0)&(y==1)).sum());discord=a_only+b_only
    rng=np.random.default_rng(20260928)
    boot=np.concatenate([d[rng.integers(0,n,(min(200,bootstrap-i),n))].mean(axis=1) for i in range(0,bootstrap,200)])
    return {"n":n,"endpoint":endpoint,"delta_A_minus_B":float(d.mean()),"delta_CI95":np.percentile(boot,[2.5,97.5]).tolist(),
            "A_only":a_only,"B_only":b_only,"p_exact":float(binomtest(a_only,discord,.5).pvalue) if discord else 1.}

def holm(rows):
    ordered=sorted(enumerate(rows),key=lambda p:p[1]["p_exact"]);last=0.;m=len(rows)
    for rank,(index,row) in enumerate(ordered):
        last=max(last,min(1.,(m-rank)*row["p_exact"]));rows[index]["p_holm"]=last

def main():
 manifest=read('MANIFEST.json');checked=0
 for item in manifest['files']:
  p=ROOT/item['path'];b=p.read_bytes()
  assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256'], 'Integrity mismatch: '+item['path']
  checked+=1
 configs=read('data/run_configurations.json.gz');jobs=defaultdict(list)
 for p in sorted((ROOT/'data/episodes').glob('*.jsonl.gz')):
  for item in lines(p):jobs[item['job_id']].append(item['record'])
 assert set(jobs)==set(configs)
 for job,rows in jobs.items():
  run=configs[job]['run'];cfg=run['resolved_config'];rows.sort(key=lambda r:int(r['episode']))
  assert [int(r['episode']) for r in rows]==list(range(cfg['episodes']))
  seeds=read('simulation/configs/seeds_'+cfg['bank']+'.json')[:len(rows)]
  assert [int(r['seed']) for r in rows]==seeds
  counts=Counter(r['outcome'] for r in rows)
  assert dict(counts)=={k:v for k,v in configs[job]['summary']['counts'].items() if v}
  for path,expected in run['input_hashes'].items():
   p=ROOT/'simulation'/path
   if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,'Source/model mismatch: '+path
 contrasts=read('analysis/archived_contrasts.json.gz');verified=0;complete_families=0;families=defaultdict(list)
 for item in contrasts:families[(item['analysis_source'],item['record']['family'],item['record']['endpoint'])].append(item)
 for key,items in families.items():
  recomputed=[]
  for item in items:
   expected=item['record']
   assert item['raw_available']==(expected['A'] in jobs and expected['B'] in jobs)
   if not item['raw_available']:continue
   n=expected.get('prefix_episodes',expected['n']);a=jobs[expected['A']][:n];b=jobs[expected['B']][:n];endpoint=expected['endpoint']
   if endpoint in ('disk_collision','any_disk_collision','disk_collision_while_pending'):
    field='disk_collision_while_pending' if endpoint=='disk_collision_while_pending' else 'any_disk_collision'
    a=[dict(r,outcome=endpoint if int(r[field]) else 'other') for r in a]
    b=[dict(r,outcome=endpoint if int(r[field]) else 'other') for r in b]
   got=paired(a,b,endpoint)
   for field in ('n','A_only','B_only'):assert got[field]==expected[field],(key,field,got,expected)
   for field in ('delta_A_minus_B','p_exact'):assert np.isclose(got[field],expected[field],rtol=1e-10,atol=1e-12),(key,field,got,expected)
   assert np.allclose(got['delta_CI95'],expected['delta_CI95'],rtol=0,atol=1e-12),(key,'CI',got,expected)
   recomputed.append((got,expected));verified+=1
  if len(recomputed)==len(items):
   holm([g for g,e in recomputed])
   for got,expected in recomputed:assert np.isclose(got['p_holm'],expected['p_holm'],atol=1e-12,rtol=1e-10),(key,'Holm',got,expected)
   complete_families+=1
 query_counts=Counter();injected=Counter();eventcounts=defaultdict(Counter);completed_latencies=defaultdict(list);premature=0;eventrows=0;event_episodes=set()
 for p in sorted((ROOT/'data/events').glob('*.jsonl.gz')):
  for item in lines(p):
   job=item['job_id'];episode=item['episode'];e=item['record'];key=(job,episode)
   assert int(jobs[job][episode]['seed'])==item['seed']
   event_episodes.add(key);eventrows+=1;eventcounts[job][e['event']]+=1
   if e['event']=='query':query_counts[key]+=1;injected[key]+=int(e['injected_failure'])
   if e['event']=='activate':premature+=int(e['first_activation_s']+1e-9<e['ready_s'])
   if e['event']=='reply_completed':completed_latencies[job].append(e['latency_s'])
 expected_event_episodes=0
 for job,meta in configs.items():
  is_live=meta['archive_group']=='live';is_virtual=meta['archive_group']=='final' and meta['run']['resolved_config']['execution']=='virtual'
  if not (is_live or is_virtual):continue
  expected_event_episodes+=len(jobs[job])
  for row in jobs[job]:
   key=(job,int(row['episode']))
   assert query_counts[key]==int(row['queries']),(key,'queries')
   assert injected[key]==int(row['injected_failures']),(key,'injected')
   # Zero-query logs are empty, so the episode need not occur in merged events.
   assert int(row['queries'])==0 or key in event_episodes
 assert premature==0
 live_jobs=[j for j,m in configs.items() if m['archive_group']=='live']
 live_totals={f:sum(int(r[f]) for j in live_jobs for r in jobs[j]) for f in ('queries','api_dispatches','provider_calls','injected_failures','deadline_misses')}
 live_totals['right_censored']=sum(eventcounts[j]['deadline_timeout']+eventcounts[j]['episode_cancel'] for j in live_jobs)
 assert live_totals=={'queries':92764,'api_dispatches':87216,'provider_calls':85018,'injected_failures':5548,'deadline_misses':63,'right_censored':2946},live_totals
 stepgroups=defaultdict(list)
 for item in lines(ROOT/'data/overrun_step_records.jsonl.gz'):stepgroups[(item['job_id'],int(item['record']['episode']))].append(item['record'])
 expected_overruns={(j,int(r['episode'])) for j in live_jobs for r in jobs[j] if int(r['deadline_misses'])}
 assert set(stepgroups)==expected_overruns and len(stepgroups)==3
 overrun_ticks=0;max_overrun_period=0.
 for key,steps in stepgroups.items():
  row=jobs[key[0]][key[1]];steps.sort(key=lambda r:int(r['step']))
  assert len(steps)==int(row['steps'])
  wall=np.array([float(r['wall_time_s']) for r in steps]);compute=np.array([float(r['compute_s']) for r in steps])
  ticks=int(sum(float(r['wall_time_s'])-.2*int(r['step'])>.02 for r in steps))
  assert ticks==int(row['deadline_misses'])
  assert np.isclose(compute.max(),float(row['compute_max_s']),rtol=0,atol=1e-12)
  assert np.isclose(np.diff(wall).max(),float(row['period_max_s']),rtol=0,atol=1e-12)
  overrun_ticks+=ticks;max_overrun_period=max(max_overrun_period,float(np.diff(wall).max()))
 assert overrun_ticks==63
 legacy=defaultdict(list)
 for item in lines(ROOT/'data/normalized/episode_records.jsonl.gz'):legacy[item['dataset_id']].append(item['record'])
 for key,rows in legacy.items():
  assert len(rows)==1000,(key,len(rows))
  assert len({int(r['seed']) for r in rows})==len(rows)
 legacycalls=Counter();legacymax=defaultdict(float)
 for item in lines(ROOT/'data/normalized/llm_query_records.jsonl.gz'):
  key=item['dataset_id'];legacycalls[key]+=1;legacymax[key]=max(legacymax[key],float(item['record']['decision_duration_s']))
 summary={'integrity_files_verified':checked,'jobs_verified':len(jobs),'episode_records_verified':sum(map(len,jobs.values())),
 'paired_contrasts_recomputed':verified,'complete_multiplicity_families_verified':complete_families,'unreleased_contrasts_context_only':len(contrasts)-verified,
 'event_rows_verified':eventrows,'event_log_episodes':expected_event_episodes,'premature_activations':premature,'live_totals':live_totals,
 'overrun_full_step_traces':len(stepgroups),'late_ticks_recomputed_from_steps':overrun_ticks,'maximum_overrun_tick_period_s':max_overrun_period,'normalized_datasets_verified':len(legacy),'normalized_episode_records_verified':sum(map(len,legacy.values())),
 'normalized_queries':dict(legacycalls),'normalized_max_duration_s':dict(legacymax),
 'completed_only_latency_quantiles_by_job':{j:np.percentile(v,[50,95,99,100]).tolist() for j,v in completed_latencies.items()},
 'note':'Subset audit. Incomplete multiplicity families retain archived context; their adjusted p values are not recalculated over a reduced family.'}
 (ROOT/'audit_result.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
 print('PASS:',len(jobs),'jobs;',summary['episode_records_verified'],'episodes;',verified,'paired contrasts;',eventrows,'events. No API calls or simulations.')
if __name__=='__main__':main()
