import math,time
import numpy as np
from baseline_eval_suite import (DEFAULT_DECISION,build_nav_context_from_obs,sanitize_decision,RuleBasedController,
                                 APFReactiveController,CBFStyleController,FormalCBFController)
from .models import Predictor
from .timing import Scheduler

def context(obs,step,episode,cfg):
    ctx=build_nav_context_from_obs(obs,step,episode)
    near=float(obs[6:].min());very=float(cfg.get("very_close_threshold",-.2));close=float(cfg.get("close_threshold",0))
    if not very<close:raise ValueError("very_close_threshold must be less than close_threshold")
    ctx["proximity_category"]="very_close" if near<very else "close" if near<close else "comfortable"
    change=float(obs[5])*100;eps=float(cfg.get("progress_threshold_m",.01))
    ctx["progress_trend"]="getting_closer" if change < -eps else "getting_farther" if change > eps else "stagnant"
    return ctx
def safe(ctx):return ctx["proximity_category"]=="comfortable" and ctx["progress_trend"]=="getting_closer"

def modulate(base,decision,ctx,cfg):
    orig=np.asarray(base,dtype=float);d=sanitize_decision(decision,ctx)
    if cfg.get("gate",True) and safe(ctx):return orig.astype(np.float32)
    physical=cfg.get("mapping","legacy") in ("physical","scale_only")
    stop_fixed=cfg.get("mapping","legacy") in ("physical","stop_only")
    if d["mode"]=="stop":return np.array([-1.,0.] if stop_fixed else [0.,0.],dtype=np.float32)
    sv,sw,bias=d["speed_scale"],d["turn_scale"],d["turn_bias"]
    if cfg.get("rules",True):
        if ctx["proximity_category"]=="very_close":sv=min(sv,.3);sw=max(sw,1.2)
        elif ctx["proximity_category"]=="close":sv=min(sv,.6)
        if ctx["progress_trend"]=="getting_farther":sw=min(max(sw,1.1),1.6)
    linear=(orig[0]+1)*sv-1 if physical else orig[0]*sv
    angular=orig[1]*sw+bias
    return np.clip([np.clip(linear,orig[0]-.3,orig[0]+.3),np.clip(angular,orig[1]-.5,orig[1]+.5)],-1,1).astype(np.float32)

DWA_SETTINGS=[
 {"name":"balanced","heading_weight":.4,"goal_weight":2.,"speed_weight":.6,"clearance_weight":.12},
 {"name":"clearance","heading_weight":.4,"goal_weight":2.,"speed_weight":.4,"clearance_weight":.24},
 {"name":"progress","heading_weight":.6,"goal_weight":2.5,"speed_weight":.9,"clearance_weight":.08}]

class DWA:
    """Sensor-based DWA: reachable velocities, circular footprint, sampled rollout, stopping admissibility.
    No ground-truth map, global planner, PPO action, or future obstacle trajectory is available to this controller.
    """
    def __init__(self,setting=None):
        self.setting=setting or DWA_SETTINGS[0];self.rejections=0
    def command(self,obs,initial=False):
        dt=.2;v0=0. if initial else .11*(float(obs[2])+1);w0=0. if initial else 2.84*float(obs[3])
        acc=.30;wacc=1.5;radius=.17;margin=.02
        vs=np.linspace(max(0,v0-acc*dt),min(.22,v0+acc*dt),7)
        ws=np.linspace(max(-2.84,w0-wacc*dt),min(2.84,w0+wacc*dt),17)
        v,w=np.meshgrid(vs,ws);v=v.ravel();w=w.ravel()
        xy=np.zeros((len(v),2));yaw=np.zeros(len(v))
        ranges=.16+.42*(np.clip(np.asarray(obs[6:],dtype=float),-1,1)+1)
        angles=np.linspace(0,2*np.pi,len(ranges),endpoint=False)
        obstacles=np.stack([ranges*np.cos(angles),ranges*np.sin(angles)],axis=1)[ranges<.995]
        clearance=np.full(len(v),1.)
        for _ in range(8):
            xy+=.2*v[:,None]*np.stack([np.cos(yaw),np.sin(yaw)],axis=1);yaw+=.2*w
            if len(obstacles):
                distances=np.sqrt(((xy[:,None,:]-obstacles[None,:,:])**2).sum(axis=2)).min(axis=1)-radius
                clearance=np.minimum(clearance,distances)
        # A conservative sufficient stopping-clearance check within the observed rollout.
        admissible=clearance>=v*v/(2*acc)+margin
        self.rejections+=int((~admissible).sum())
        if not np.any(admissible):return np.array([-1.,0.],dtype=np.float32),{"dwa_no_admissible":1}
        distance=float(obs[4])*100;heading=math.atan2(float(obs[0]),float(obs[1]))
        goal=np.array([distance*math.cos(heading),distance*math.sin(heading)])
        endpoint=np.linalg.norm(xy-goal,axis=1)
        heading_error=np.abs(np.arctan2(np.sin(heading-yaw),np.cos(heading-yaw)))
        c=self.setting
        score=c["goal_weight"]*endpoint+c["heading_weight"]*heading_error-c["speed_weight"]*v/.22+c["clearance_weight"]/np.maximum(clearance+.02,.01)
        score[~admissible]=np.inf;idx=int(score.argmin())
        return np.array([v[idx]/.11-1,w[idx]/2.84],dtype=np.float32),{"dwa_no_admissible":0}

class Controller:
    def __init__(self,cfg,policies,event_sink):
        self.cfg=cfg;self.policies=policies;self.method=cfg["method"];self.event_sink=event_sink
        self.latest_obs=None;self.last=np.array([-1.,0.]);self.delta=np.zeros(2)
        self.source=None;self.predictor=None;self.scheduler=None
        if self.method in ("mlp","independent"):self.predictor=Predictor(self.method).predict
        elif self.method=="rule":self.source=RuleBasedController(1);self.predictor=self.source.infer_decision
        elif self.method=="apf":self.source=APFReactiveController();self.predictor=self.source.infer_decision
        elif self.method=="gate":self.predictor=lambda ctx:dict(DEFAULT_DECISION)
        elif self.method=="gemini":pass
        elif self.method=="cbf":self.source=CBFStyleController()
        elif self.method=="cbf_qp":
            self.source=FormalCBFController()
            self.source.lookahead_m=float(cfg.get("lookahead_m",.08))
            self.source.safe_radius_m=float(cfg.get("cbf_radius_m",.12))
            self.source.alpha=float(cfg.get("cbf_alpha",3))
        elif self.method=="dwa":self.source=DWA(cfg.get("dwa_setting"))
        elif self.method not in ("ppo","fsrl"):raise ValueError(self.method)
        if self.method in ("mlp","independent","rule","apf","gate","gemini"):
            self.scheduler=Scheduler(cfg,self.predictor,event_sink)
    def reset(self,seed,episode):
        self.episode=episode;self.delta=np.zeros(2);self.last=np.array([-1.,0.]);self.decision=dict(DEFAULT_DECISION)
        if hasattr(self.source,"reset_episode"):self.source.reset_episode()
        if self.scheduler:self.scheduler.reset(seed,episode)
    def action(self,obs,step):
        base=(np.array([-1.,0.],dtype=np.float32) if self.method=="dwa" else self.policies.fsrl_action(obs,self.cfg.get("checkpoint","final")) if self.method=="fsrl" else self.policies.ppo_action(obs))
        ctx=context(obs,step,self.episode,self.cfg);self.latest_obs=np.asarray(obs).copy()
        decision=dict(DEFAULT_DECISION);emergency=False
        if self.scheduler:
            if self.method=="apf":self.source.last_obs=self.latest_obs
            decision=self.scheduler.tick(step,ctx,safe(ctx))
            action=modulate(base,decision,ctx,self.cfg)
            emergency=self.cfg.get("fallback","hold")=="emergency" and (self.scheduler.pending is not None or self.scheduler.last_failure)
            if emergency:
                very=ctx["proximity_category"]=="very_close"
                if very:
                    decision={"mode":"stop","speed_scale":0.,"turn_scale":1.,"turn_bias":0.}
                    action=np.array([-1.,0.] if self.cfg.get("mapping")=="physical" else [0.,0.],dtype=np.float32)
                elif ctx["proximity_category"]=="close":
                    decision={"mode":"cautious","speed_scale":.3,"turn_scale":1.2,"turn_bias":0.}
                    action=modulate(base,decision,ctx,{**self.cfg,"gate":False})
        elif self.method in ("cbf","cbf_qp"):
            if step%int(self.cfg.get("K",1))==0:
                self.source.maybe_update(step,obs,self.episode)
                corrected=self.source.apply_decision(base,DEFAULT_DECISION)
                self.delta=np.asarray(corrected)-base
            action=np.clip(base+self.delta,-1,1)
        elif self.method=="dwa":
            if step%int(self.cfg.get("K",1))==0:self.last,_=self.source.command(obs,initial=step==0)
            action=self.last.copy()
        else:action=base.copy()
        self.decision=decision
        diag={"base_a0":float(base[0]),"base_a1":float(base[1]),"a0":float(action[0]),"a1":float(action[1]),
              "mode":decision["mode"],"speed_scale":decision["speed_scale"],"turn_scale":decision["turn_scale"],"turn_bias":decision["turn_bias"],
              "pending":int(self.scheduler is not None and self.scheduler.pending is not None),"emergency":int(emergency),
              "query_age_s":float(self.scheduler.now(step)-self.scheduler.query_time) if self.scheduler and self.scheduler.query_time is not None else -1.,
              "min_lidar_norm":float(obs[6:].min()),"proximity":ctx["proximity_category"],"progress":ctx["progress_trend"]}
        return np.asarray(action,dtype=np.float32),diag
    def finish(self,step):
        if self.scheduler:self.scheduler.finish(step)
