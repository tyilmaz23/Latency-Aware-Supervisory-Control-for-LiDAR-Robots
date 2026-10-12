import argparse,csv,gzip,json,os,time,traceback
from pathlib import Path
import numpy as np
from .common import ROOT,read_json,atomic_json,csv_write,digest,input_hashes,versions,scene
from .environment import RevisionEnv
from .models import Policies
from .controllers import Controller
from .timing import BudgetExceeded

def run(job,outdir,api_cap=100000,max_steps=None):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    config=dict(job["config"]);original_config=dict(config)
    if config.get("dwa_setting")=="validation_selected":
        selected=read_json(outdir.parent/"dwa_selection.json")
        config["dwa_setting"]=selected["setting"];config["dwa_validation_selection"]=selected
    seeds=read_json(ROOT/("configs/seeds_"+config["bank"]+".json"))[:config["episodes"]]
    if len(seeds)!=config["episodes"]:raise RuntimeError("Requested sample exceeds frozen bank length")
    metadata={"job":job,"resolved_config":config,"seed_hash":digest(seeds),"input_hashes":input_hashes(),"versions":versions(),
              "max_steps_override":max_steps,"primary_checkpoint_rule":"end_of_budget_final; FSRL best is predefined sensitivity only"}
    fingerprint=digest(metadata)
    snapshot=outdir/"run.json"
    if snapshot.exists():
        previous=read_json(snapshot)
        if previous["fingerprint"]!=fingerprint:raise RuntimeError("Resume refused: code/config/assets/dependencies changed. Use a new output directory.")
    else:atomic_json(snapshot,{"fingerprint":fingerprint,**metadata})
    status_path=outdir/"status.json"
    status={"state":"running","job":job["id"],"completed_episodes":0,"planned_episodes":len(seeds)}
    config["api_ledger"]=str(outdir.parent/"api_ledger.json");config["api_cap"]=api_cap
    policies=Policies();env=RevisionEnv(config["environment"])
    if max_steps is not None:env.max_steps=int(max_steps)
    controller=Controller(config,policies,lambda item:events.write(json.dumps(item,allow_nan=False)+"\n"))
    committed={}
    for f in sorted((outdir/"episodes").glob("*.json")) if (outdir/"episodes").exists() else []:
        record=read_json(f);committed[int(record["episode"])]=record
    if any(seeds[i]!=int(row["seed"]) for i,row in committed.items()):raise RuntimeError("Resume seed mismatch")
    status["completed_episodes"]=len(committed);atomic_json(status_path,status)
    started=time.perf_counter()
    try:
        for episode,seed in enumerate(seeds):
            if episode in committed:continue
            episode_dir=outdir/"episodes";trace_dir=outdir/"traces";episode_dir.mkdir(exist_ok=True);trace_dir.mkdir(exist_ok=True)
            attempts=list(trace_dir.glob("ep_%04d_attempt*.csv.gz"%episode));attempt=len(attempts)+1
            trace=trace_dir/("ep_%04d_attempt%02d.csv.gz"%(episode,attempt))
            event_path=trace_dir/("ep_%04d_attempt%02d.events.jsonl.gz"%(episode,attempt))
            with gzip.open(event_path,"wt",encoding="utf-8") as events,gzip.open(trace,"wt",encoding="utf-8",newline="") as trace_file:
                obs,_=env.reset(seed=int(seed));initial=scene(env);initial_hash=env.scene_hash
                atomic_json(outdir/"scenarios"/("ep_%04d.json"%episode),initial)
                controller.reset(seed,episode)
                total_reward=0.;path=0.;minimum=float(env.true_clearance);ground_collision=False;pending_collision=False
                increases=0;stop_labels=0;stop_nonzero=0;deadline_misses=0;computations=[];periods=[];previous_tick=None
                writer=None;wall_start=time.perf_counter();steps=0
                for step in range(env.max_steps+2):
                    if (outdir.parent/"STOP_REQUESTED").exists():raise KeyboardInterrupt("Graceful stop requested")
                    if config["execution"]=="wallclock":
                        remaining=wall_start+step*.2-time.perf_counter()
                        if remaining>0:time.sleep(remaining)
                    tick=time.perf_counter()
                    if previous_tick is not None:periods.append(tick-previous_tick)
                    previous_tick=tick
                    t0=time.perf_counter();action,diag=controller.action(obs,step);compute=time.perf_counter()-t0;computations.append(compute)
                    if config["execution"]=="wallclock" and tick-wall_start-step*.2>.02:deadline_misses+=1
                    previous_pose=env.agent_pos.copy()
                    obs,reward,terminated,truncated,info=env.step(action)
                    commanded_v=.11*(float(action[0])+1);base_v=.11*(diag["base_a0"]+1)
                    eligible=config["method"] in ("mlp","independent","gate","rule","apf","gemini")
                    increases+=int(eligible and commanded_v>base_v+1e-7)
                    stop_labels+=int(diag["mode"]=="stop");stop_nonzero+=int(diag["mode"]=="stop" and commanded_v>1e-7)
                    clearance=float(info["true_clearance_m"]);minimum=min(minimum,clearance)
                    ground_collision|=bool(info["ground_truth_disk_collision"])
                    pending_collision|=bool(diag["pending"] and info["ground_truth_disk_collision"])
                    path+=float(np.linalg.norm(env.agent_pos[:2]-previous_pose[:2]));total_reward+=float(reward);steps=step+1
                    row={"episode":episode,"seed":seed,"step":step,"sim_time_s":step*.2,"wall_time_s":tick-wall_start,
                         "x":float(env.agent_pos[0]),"y":float(env.agent_pos[1]),"heading":float(env.agent_pos[2]),
                         "goal_distance":float(np.linalg.norm(env.agent_pos[:2]-env.target_pos)),**diag,
                         "command_v":commanded_v,"command_omega":2.84*float(action[1]),
                         "executed_v":.11*(float(env.actual_command[0])+1),"executed_omega":2.84*float(env.actual_command[1]),
                         "true_clearance":clearance,"disk_collision":int(info["ground_truth_disk_collision"]),
                         "goal_condition":int(info["goal_condition"]),"compute_s":compute}
                    if writer is None:writer=csv.DictWriter(trace_file,fieldnames=list(row));writer.writeheader()
                    writer.writerow(row)
                    if terminated or truncated:break
                    if config["execution"]=="blocked":
                        remaining=.2-(time.perf_counter()-tick)
                        if remaining>0:time.sleep(remaining)
                else:raise RuntimeError("Environment failed to terminate at its declared counter threshold")
                controller.finish(steps)
            reason=info.get("reason","")
            outcome={"target_reached":"reached","collision":"collision","max_steps_reached":"timeout"}.get(reason)
            if outcome is None:raise RuntimeError("Unrecognized outcome reason: "+reason)
            metrics=controller.scheduler.metrics if controller.scheduler else {}
            result={"episode":episode,"seed":int(seed),"scene_hash":initial_hash,"outcome":outcome,"steps":steps,
                    "reward":total_reward,"path_length_m":path,"duration_sim_s":steps*.2,"duration_wall_s":time.perf_counter()-wall_start,
                    "initial_distance_m":float(np.linalg.norm(np.asarray(initial["agent"])[:2]-np.asarray(initial["goal"]))),
                    "final_distance_m":float(np.linalg.norm(env.agent_pos[:2]-env.target_pos)),
                    "minimum_true_clearance_m":minimum,"any_disk_collision":int(ground_collision),"disk_collision_while_pending":int(pending_collision),
                    "simultaneous_goal_collision":int(info["goal_condition"] and info["ground_truth_disk_collision"]),
                    "speed_increase_steps":increases,"stop_label_steps":stop_labels,"stop_nonzero_steps":stop_nonzero,
                    "deadline_misses":deadline_misses,"compute_mean_s":float(np.mean(computations)),
                    "compute_p95_s":float(np.percentile(computations,95)),"compute_max_s":float(max(computations)),
                    "period_p95_s":float(np.percentile(periods,95)) if periods else 0.,
                    "period_max_s":float(max(periods)) if periods else 0.,
                    "trace":str(trace.relative_to(outdir)),"events":str(event_path.relative_to(outdir)),**metrics}
            atomic_json(episode_dir/("ep_%04d.json"%episode),result);committed[episode]=result
            status.update(completed_episodes=len(committed),latest_episode=episode,elapsed_this_invocation_s=time.perf_counter()-started)
            atomic_json(status_path,status)
            if len(committed)%10==0 or len(committed)==len(seeds):
                print(job["id"],str(len(committed))+"/"+str(len(seeds)),"latest",outcome,flush=True)
        ordered=[committed[i] for i in range(len(seeds))]
        csv_write(outdir/"episodes.csv",ordered)
        counts={key:sum(r["outcome"]==key for r in ordered) for key in ("reached","collision","timeout")}
        total_calls=sum(r.get("api_dispatches",0) for r in ordered)
        success_replies=sum(r.get("activated",0)+r.get("discarded",0) for r in ordered)
        summary={"episodes":len(ordered),"counts":counts,"rates":{k:v/len(ordered) for k,v in counts.items()},
                 "seed_hash":digest(seeds),"mean_steps":float(np.mean([r["steps"] for r in ordered])),
                 "mean_path_m":float(np.mean([r["path_length_m"] for r in ordered])),
                 "api_dispatches":total_calls,"observed_provider_completed":sum(r.get("provider_calls",0) for r in ordered),"successful_reply_events":success_replies,
                 "backend_valid":config["method"]!="gemini" or (total_calls>0 and success_replies>0)}
        atomic_json(outdir/"summary.json",summary)
        status["state"]="complete" if summary["backend_valid"] else "complete_backend_unavailable"
        atomic_json(status_path,status);return summary
    except BaseException as exc:
        try:controller.finish(steps if "steps" in locals() else 0)
        except Exception:pass
        status.update(state="budget_blocked" if isinstance(exc,BudgetExceeded) else "interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",
                      error_type=type(exc).__name__,error_message=str(exc),completed_episodes=len(committed))
        atomic_json(status_path,status)
        raise
    finally:env.close()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--job",required=True);parser.add_argument("--outdir",required=True)
    parser.add_argument("--api-cap",type=int,default=100000);parser.add_argument("--max-steps",type=int)
    args=parser.parse_args();run(read_json(args.job),args.outdir,args.api_cap,args.max_steps)
if __name__=="__main__":main()
