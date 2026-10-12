"""Strict paired analysis; no pilot output can silently enter manuscript tables."""
import argparse,gzip,json,math
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from .common import ROOT,read_json,atomic_json,csv_write
from .plan import build_plan

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

def events_for(out,rows):
    for r in rows:
        with gzip.open(out/r["events"],"rt",encoding="utf-8") as handle:
            for line in handle:yield json.loads(line)

def analyze(outdir="outputs"):
    out=Path(outdir).resolve();manifest=read_json(out/"queue_manifest.json")
    plan=list(manifest["jobs"])
    if (out/"hardware_matched_manifest.json").exists():plan+=read_json(out/"hardware_matched_manifest.json")["jobs"]
    complete={};coverage=[];tables=[];latency=[]
    for job in plan:
        folder=out/job["id"];status=read_json(folder/"status.json") if (folder/"status.json").exists() else {"state":"not_started"}
        valid=status["state"]=="complete"
        coverage.append({"job":job["id"],"groups":",".join(job["groups"]),"state":status["state"],"planned":job["config"]["episodes"],"completed":status.get("completed_episodes",0)})
        if not valid:continue
        rows=[read_json(p) for p in sorted((folder/"episodes").glob("*.json"))]
        if len(rows)!=job["config"]["episodes"]:raise ValueError("Incomplete committed row count")
        complete[job["id"]]=(job,rows)
        n=len(rows);row={"job":job["id"],"groups":",".join(job["groups"]),"method":job["config"]["method"],"n":n}
        for endpoint in ("reached","collision","timeout"):
            k=sum(r["outcome"]==endpoint for r in rows);lo,hi=wilson(k,n)
            row.update({endpoint+"_count":k,endpoint+"_rate":k/n,endpoint+"_CI_low":lo,endpoint+"_CI_high":hi})
        for key in ("steps","path_length_m","compute_mean_s","compute_p95_s","speed_increase_steps","stop_nonzero_steps","any_disk_collision","disk_collision_while_pending","deadline_misses"):
            row[key+"_mean"]=float(np.mean([r[key] for r in rows]))
        row["api_dispatches"]=sum(r.get("api_dispatches",0) for r in rows);tables.append(row)
        durations=[];censored=0;early=0;failures=0;queries=0;injections=0;burst=0;longest=0;activated=0;applied=False
        last_episode=None;tail_queries=0
        for event in events_for(folder,rows):
            if event["episode"]!=last_episode:burst=0;last_episode=event["episode"]
            if event["event"]=="query" and event.get("planned_delay_s",0)>=20:tail_queries+=1
            if event["event"]=="query":queries+=1;injections+=int(event.get("injected_failure",False))
            if event["event"] in ("deadline_timeout","episode_cancel"):censored+=1
            if event["event"]=="activate":
                activated+=1;early+=int(event["first_activation_s"]+1e-7<event["ready_s"])
            if event["event"]=="reply_completed":
                durations.append(event["latency_s"]);bad=event.get("request_failed",False)
                failures+=int(bad);burst=burst+1 if bad else 0;longest=max(longest,burst)
            if event["event"]=="deadline_timeout":burst+=1;longest=max(longest,burst)
        latency.append({"job":job["id"],"queries":queries,"injected_tail_queries":tail_queries,"injected_failure_queries":injections,"reply_failures":failures,"censored_requests":censored,
                        "activated":activated,"premature_activations":early,"longest_observed_failure_chain":longest,
                        "uncensored_reply_p50_s":float(np.percentile(durations,50)) if durations else None,
                        "uncensored_reply_p95_s":float(np.percentile(durations,95)) if durations else None,
                        "uncensored_reply_p99_s":float(np.percentile(durations,99)) if durations else None,
                        "uncensored_reply_max_s":max(durations) if durations else None})
    # Only match policies with identical scene generation, seed bank, and scene hashes.
    contrasts=[];seen=set()
    def compare(aid,bid,family,prefix=False):
        if aid==bid or (aid,bid,family) in seen or aid not in complete or bid not in complete:return
        ja,a=complete[aid];jb,b=complete[bid]
        if ja["config"]["bank"]!=jb["config"]["bank"]:return
        if prefix:
            common={r["seed"] for r in a}&{r["seed"] for r in b};a=[r for r in a if r["seed"] in common];b=[r for r in b if r["seed"] in common]
        if not a or {r["seed"] for r in a}!={r["seed"] for r in b}:return
        seen.add((aid,bid,family))
        for endpoint in ("reached","collision","timeout"):
            result=paired(a,b,endpoint);contrasts.append({"A":aid,"B":bid,"family":family+"_"+endpoint,**result})
    for aid,(ja,a) in complete.items():
        cfg=ja["config"]
        if cfg["method"]!="ppo":
            candidates=[bid for bid,(jb,b) in complete.items() if jb["config"]["method"]=="ppo" and jb["config"]["bank"]==cfg["bank"] and jb["config"]["environment"]==cfg["environment"]]
            if candidates:compare(aid,candidates[0],"vs_PPO_"+next(iter(ja["groups"])),prefix=True)
    for group in ("ablation","mapping","virtual","live","checkpoint"):
        entries=[(jid,j["config"]) for jid,(j,r) in complete.items() if group in j["groups"]]
        allowed={"ablation":("gate","rules","K","mapping"),"mapping":("mapping",),"virtual":("fallback","fault","request_deadline_s"),"live":("execution","fallback","fault","K"),"checkpoint":("checkpoint",)}[group]
        for i,(aid,a) in enumerate(entries):
            for bid,b in entries[i+1:]:
                differing=[k for k in set(a)|set(b) if a.get(k)!=b.get(k)]
                if len(differing)==1 and differing[0] in allowed:compare(aid,bid,group+"_"+differing[0])
    # Primary checkpoint sensitivity also compares best/final across native rows.
    for aid,(ja,a) in complete.items():
        if ja["config"].get("checkpoint")=="best":
            cfg={k:v for k,v in ja["config"].items() if k!="checkpoint"}
            for bid,(jb,b) in complete.items():
                if cfg=={k:v for k,v in jb["config"].items() if k!="checkpoint"}:compare(aid,bid,"checkpoint")
    for family in set(r["family"] for r in contrasts):holm([r for r in contrasts if r["family"]==family])
    report=out/"analysis";report.mkdir(exist_ok=True)
    csv_write(report/"job_coverage.csv",coverage);csv_write(report/"outcomes.csv",tables);csv_write(report/"latency_and_faults.csv",latency)
    if contrasts:csv_write(report/"paired_contrasts.csv",[{**r,"delta_CI95":json.dumps(r["delta_CI95"])} for r in contrasts])
    atomic_json(report/"paired_contrasts.json",contrasts)
    hardware_files=list((out/"hardware").glob("trials/*.json"))
    hardware_analysis(out,report)
    gaps={"pilot_data":manifest["pilot"],"jobs_complete":len(complete),"jobs_planned":len(plan),"all_simulation_complete":len(complete)==len(plan),
          "premature_activations":sum(r["premature_activations"] for r in latency),"hardware_records":len(hardware_files),
          "remaining_nonautomatic_items":["multi-layout hardware trials need real execution and contact labels",
          "measured simulation-to-real calibration requires external measurements","public repository/DOI must be published separately",
          "formal safety/stability proof is not supplied; claims must remain empirical",
          "old-computer K10/20-trial records accepted as genuine; retrieval is useful for public reproducibility"]}
    atomic_json(report/"coverage.json",gaps)
    text="Pilot: "+str(manifest["pilot"])+"; completed jobs: "+str(len(complete))+"/"+str(len(plan))+"\n"
    text+="Completion and collision use separate endpoints. Non-significant collision tests do not establish equivalence or safety.\n"
    text+="Latency quantiles above exclude cancelled/deadline-censored requests; do not describe them as full-tail quantiles.\n"
    text+="Hardware aborts must not be removed from the denominator without disclosure.\n"
    (report/"INTERPRETATION.txt").write_text(text,encoding="utf-8")
    print(text);return gaps

def hardware_analysis(out,report):
    files=list((out/"hardware/trials").glob("*.json"))
    if not files:return
    records=[read_json(p) for p in files];aggregates=[];contrasts=[]
    for layout in sorted({r["layout"] for r in records}):
        for method,k in sorted({(r["method"],r["K"]) for r in records if r["layout"]==layout}):
            rows=[r for r in records if (r["layout"],r["method"],r["K"])==(layout,method,k)]
            n=len(rows);data={"layout":layout,"method":method,"K":k,"attempted_trials":n,
                            "missing_contact_labels":sum(r.get("physical_contact_label") not in ("YES","NO") for r in rows),
                            "invalid_backend":sum(not r["backend_valid"] for r in rows)}
            for label in ("reached","collision","timeout","operator_abort","protocol_abort"):data[label]=sum(r["outcome"]==label for r in rows)
            aggregates.append(data)
            if method=="ppo":continue
            baseline={r["block"]:r for r in records if (r["layout"],r["method"])==(layout,"ppo")}
            common=[r for r in rows if r["block"] in baseline]
            if not common:continue
            a=[{"seed":r["block"],"scene_hash":r["scene_hash"],"outcome":r["outcome"]} for r in common]
            b=[{"seed":r["block"],"scene_hash":baseline[r["block"]]["scene_hash"],"outcome":baseline[r["block"]]["outcome"]} for r in common]
            for endpoint in ("reached","collision"):
                contrasts.append({"layout":layout,"method":method,"K":k,"family":"hardware_"+endpoint,**paired(a,b,endpoint)})
    for endpoint in ("reached","collision"):holm([r for r in contrasts if r["endpoint"]==endpoint])
    csv_write(report/"hardware_outcomes.csv",aggregates)
    atomic_json(report/"hardware_paired_contrasts.json",contrasts)

def main():
    p=argparse.ArgumentParser();p.add_argument("--outdir",default="outputs");a=p.parse_args();analyze(a.outdir)
if __name__=="__main__":main()
