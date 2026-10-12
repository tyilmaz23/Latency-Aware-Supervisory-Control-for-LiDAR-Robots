import json,math,os,queue,re,subprocess,sys,threading,time
from pathlib import Path
import numpy as np
from baseline_eval_suite import build_supervisor_prompt,sanitize_decision,DEFAULT_DECISION
from .common import ROOT,read_json,atomic_json

class BudgetExceeded(RuntimeError):pass
class Worker:
    def __init__(self):self.process=None;self.messages=queue.Queue();self.ready=False;self.generation=0
    def start(self):
        self.stop();self.generation+=1;generation=self.generation
        flags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0
        self.process=subprocess.Popen([sys.executable,"-u","-m","revision_suite.api_worker"],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,encoding="utf-8",bufsize=1,creationflags=flags)
        process=self.process
        def receive():
            try:
                for line in process.stdout:
                    item=json.loads(line);item["received_at"]=time.perf_counter();item["generation"]=generation;self.messages.put(item)
            except (OSError,ValueError):pass
        threading.Thread(target=receive,daemon=True).start()
    def poll(self):
        items=[]
        while True:
            try:item=self.messages.get_nowait()
            except queue.Empty:break
            if item["generation"]!=self.generation:continue
            if item["event"]=="worker_ready":self.ready=True
            else:items.append(item)
        return items
    def warmup(self):
        self.start();deadline=time.perf_counter()+30
        while not self.ready:
            self.poll()
            if self.process.poll() is not None:raise RuntimeError("API worker failed to start")
            if time.perf_counter()>deadline:raise RuntimeError("API worker startup timeout")
            time.sleep(.01)
    def send(self,item):
        if not self.ready:raise RuntimeError("Worker not ready")
        self.process.stdin.write(json.dumps(item)+"\n");self.process.stdin.flush()
    def stop(self):
        self.ready=False
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
            for f in [self.process.stdin,self.process.stdout]:
                if f:
                    try:f.close()
                    except OSError:pass
        self.process=None

class Faults:
    def __init__(self,profile,seed):
        self.profile=profile;self.rng=np.random.default_rng(np.random.SeedSequence([int(seed),99231]))
        self.pool=np.asarray(read_json(ROOT/"configs/empirical_latency_seconds.json"))
        self.early_index=int(self.rng.integers(1,4))
    def draw(self,index,virtual):
        profile=self.profile;base=float(self.rng.choice(self.pool)) if virtual and profile!="zero" else 0.
        extra=0.;forced=False
        if profile=="tail20" and index==self.early_index:extra=20.
        if profile=="tail161" and index==self.early_index:extra=161.34
        if profile=="burst3" and self.early_index<=index<self.early_index+3:forced=True
        if profile=="burst5" and self.early_index<=index<self.early_index+5:forced=True
        if profile=="iid_fail" and self.rng.random()<.10:forced=True
        if profile not in ("zero","empirical","tail20","tail161","burst3","burst5","iid_fail"):raise ValueError(profile)
        if forced and not virtual:extra+=15.
        return base+extra,forced

class Scheduler:
    """The ONLY active-decision mutation occurs after a ready reply is polled."""
    def __init__(self,config,predictor=None,event_sink=None):
        self.cfg=config;self.predictor=predictor;self.sink=event_sink or (lambda x:None)
        self.wall=config.get("execution","instant") in ("wallclock","blocked");self.virtual=config.get("execution")=="virtual"
        self.worker=Worker() if self.wall else None
        self.serial=0;self.pending=None;self.active=dict(DEFAULT_DECISION);self.active_since=None
        self.query_time=None;self.last_failure=False;self.first=True;self.resets=0
    def reset(self,seed,episode):
        if self.worker:self.worker.warmup()
        self.pending=None;self.active=dict(DEFAULT_DECISION);self.active_since=None;self.query_time=None
        self.faults=Faults(self.cfg.get("fault","zero"),seed);self.episode=episode;self.index=0;self.last_failure=False
        self.started=time.perf_counter()
        self.metrics={"queries":0,"api_dispatches":0,"provider_calls":0,"request_failures":0,"parse_failures":0,"activated":0,"discarded":0,
                      "deadline_timeouts":0,"skipped_pending":0,"skipped_worker":0,"injected_failures":0,"expired_active":0}
    def event(self,event,**values):
        item=dict(event=event,episode=self.episode,**values);self.sink(item)
    def now(self,step):
        return time.perf_counter()-self.started if self.wall else step*.2
    def reserve(self):
        ledger=Path(self.cfg["api_ledger"]);payload=read_json(ledger) if ledger.exists() else {"reserved_dispatches":0}
        cap=int(self.cfg.get("api_cap",100000))
        if payload["reserved_dispatches"]>=cap:raise BudgetExceeded("API dispatch cap reached; increase --api-cap to continue completed-episode resume.")
        payload["reserved_dispatches"]+=1;atomic_json(ledger,payload)
    def parse(self,raw,ctx):
        match=re.search(r"\{.*\}",raw,re.S)
        value=json.loads(match.group(0) if match else raw)
        if not isinstance(value,dict):raise ValueError("not_an_object")
        if not all(np.isfinite(float(value.get(k,DEFAULT_DECISION[k]))) for k in ("speed_scale","turn_scale","turn_bias")):raise ValueError("non_finite_coefficient")
        result=sanitize_decision(value,ctx)
        if not all(np.isfinite(float(result[k])) for k in ("speed_scale","turn_scale","turn_bias")):raise ValueError("non_finite_coefficient")
        return result
    def poll(self,now,step,ctx,safe):
        replies=self.worker.poll() if self.worker else []
        if self.virtual and self.pending and now+1e-9>=self.pending["ready"]:
            replies.append(self.pending["reply"])
        for reply in replies:
            if self.pending is None or reply["id"]!=self.pending["id"]:
                self.event("stale_unmatched_reply",id=reply["id"]);continue
            pending=self.pending;self.pending=None
            latency=now-pending["initiated"]
            self.metrics["provider_calls"]+=int(reply.get("provider_called",False))
            self.event("reply_completed",id=reply["id"],step=step,latency_s=latency,duration_s=reply.get("duration_s"),raw=reply.get("raw",""),usage=reply.get("usage",{}),model_version=reply.get("model_version",""),provider_called=reply.get("provider_called",False),right_censored=False,request_failed=reply["request_failed"])
            if reply["request_failed"]:
                self.last_failure=True;self.metrics["request_failures"]+=1
                self.event("request_failure",id=reply["id"],step=step,latency_s=latency,
                           injected=pending["forced"],provider_called=reply.get("provider_called",False),error=reply.get("error",""))
                continue
            try:decision=self.parse(reply["raw"],pending["context"])
            except (ValueError,TypeError,KeyError):
                self.metrics["parse_failures"]+=1;self.last_failure=True
                self.event("parse_failure",id=reply["id"],step=step,latency_s=latency,raw=reply["raw"]);continue
            ttl=self.cfg.get("response_ttl_s",20.)
            discard=(safe and self.cfg.get("gate",True)) or (ttl is not None and latency>float(ttl))
            if discard:
                self.metrics["discarded"]+=1
                self.event("discard",id=reply["id"],step=step,latency_s=latency,reason="comfortable_progress" if safe else "stale_age")
            else:
                self.active=decision;self.active_since=now;self.query_time=pending["initiated"];self.last_failure=False
                self.metrics["activated"]+=1
                self.event("activate",id=reply["id"],step=step,first_activation_s=now,query_s=pending["initiated"],
                           ready_s=reply.get("duration_s",0)+pending["initiated"],latency_s=latency,decision=decision,
                           nav_context_at_query=pending["context"],nav_context_at_activation=ctx)
        deadline=self.cfg.get("request_deadline_s",20.)
        if self.pending and deadline is not None and now-self.pending["initiated"]>=float(deadline):
            pending=self.pending;self.pending=None;self.last_failure=True;self.metrics["deadline_timeouts"]+=1
            self.event("deadline_timeout",id=pending["id"],step=step,age_s=now-pending["initiated"],right_censored=True,injected=pending["forced"])
            if self.worker:self.worker.start() # terminate ONLY the private worker; control loop never waits on HTTP.
        ttl=self.cfg.get("decision_ttl_s",20.)
        if self.query_time is not None and ttl is not None and now-self.query_time>float(ttl):
            self.active=dict(DEFAULT_DECISION);self.query_time=None;self.metrics["expired_active"]+=1
            self.event("active_expired",step=step)
    def tick(self,step,ctx,safe):
        now=self.now(step);self.poll(now,step,ctx,safe)
        if self.cfg.get("gate",True) and safe:
            self.active=dict(DEFAULT_DECISION);self.query_time=None
            return dict(self.active)
        if step%int(self.cfg.get("K",30))!=0:return dict(self.active)
        if self.pending:
            self.metrics["skipped_pending"]+=1;return dict(self.active)
        if self.worker and not self.worker.ready:
            self.metrics["skipped_worker"]+=1;return dict(self.active)
        self.index+=1;self.serial+=1;self.metrics["queries"]+=1
        delay,forced=self.faults.draw(self.index,self.virtual)
        self.metrics["injected_failures"]+=int(forced)
        query_id=self.serial
        if not self.wall and not self.virtual:
            if forced:return dict(self.active)
            self.active=self.predictor(ctx);self.query_time=now;self.metrics["activated"]+=1
            self.event("instant_update",step=step,decision=self.active,nav_context=ctx)
            return dict(self.active)
        context=dict(ctx)
        self.pending={"id":query_id,"initiated":now,"context":context,"forced":forced,"ready":now+delay}
        self.event("query",id=query_id,step=step,query_s=now,planned_delay_s=delay,injected_failure=forced,nav_context=context)
        if self.worker:
            if not forced:
                self.reserve();self.metrics["api_dispatches"]+=1
            self.worker.send({"id":query_id,"model":self.cfg.get("gemini_model","gemini-2.5-flash"),
                              "prompt":build_supervisor_prompt(context),"forced_failure":forced,
                              "extra_delay_s":delay,"http_timeout_s":self.cfg.get("http_timeout_s",15)})
        else:
            decision=self.predictor(context)
            self.pending["reply"]={"id":query_id,"raw":json.dumps(decision),"request_failed":forced,
                                    "error":"injected_failure" if forced else "","provider_called":False,"duration_s":delay}
            if delay<=0:self.poll(now,step,ctx,safe)
        if self.cfg.get("execution")=="blocked":
            while self.pending is not None:
                if self.cfg.get("api_ledger") and (Path(self.cfg["api_ledger"]).parent/"STOP_REQUESTED").exists():raise KeyboardInterrupt("Graceful stop requested")
                time.sleep(.01);self.poll(self.now(step),step,ctx,safe)
        return dict(self.active)
    def finish(self,step):
        try:
            if self.pending:self.event("episode_cancel",id=self.pending["id"],step=step,right_censored=True,age_s=self.now(step)-self.pending["initiated"])
        finally:
            self.pending=None
            if self.worker:self.worker.stop()
