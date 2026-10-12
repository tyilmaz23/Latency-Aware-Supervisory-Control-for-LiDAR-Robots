"""Predeclared experiments. Test and validation seeds are disjoint from all supplied seed banks."""
from .common import ROOT,read_json,digest
from .controllers import DWA_SETTINGS

CONDITIONS={
 "nominal":{},
 "corridor":{"env_variant":"ood_corridor"},
 "utrap":{"env_variant":"ood_u_traps"},
 "dynamic":{"env_variant":"ood_dynamic_crossing"},
 "noise005":{"lidar_noise_std":.05},
 "noise010":{"lidar_noise_std":.10},
 "drop010":{"beam_dropout_rate":.10},
 "drop030":{"beam_dropout_rate":.30}}

def build_plan(settings=None):
    settings=settings or read_json(ROOT/"configs/settings.json")
    jobs=[];by_hash={}
    def add(group,label,bank="test",env=None,episodes=None,**cfg):
        env={"env_variant":"baseline","lidar_mode":"legacy_first_wall","collision_mode":"legacy_rays","body_radius_m":.17,**(env or {})}
        config={"bank":bank,"episodes":episodes or settings["test_episodes"],"environment":env,
                "K":30,"gate":True,"rules":True,"mapping":"physical","execution":"instant","fault":"zero","fallback":"hold",
                "response_ttl_s":20.,"decision_ttl_s":20.,"request_deadline_s":20.,"gemini_model":settings["gemini_model"],
                "http_timeout_s":15.,**cfg}
        fingerprint=digest(config)
        if fingerprint in by_hash:
            job=by_hash[fingerprint]
            if group not in job["groups"]:job["groups"].append(group)
            job["aliases"].append(label)
            return job["id"]
        job={"id":label+"_"+fingerprint[:8],"groups":[group],"aliases":[],"config":config}
        jobs.append(job);by_hash[fingerprint]=job
        return job["id"]
    nval=settings["validation_episodes"]
    for method in ("ppo","fsrl"):
        add("validation","val_"+method,"validation",episodes=nval,method=method,K=1)
    add("validation","val_fsrl_best","validation",episodes=nval,method="fsrl",K=1,checkpoint="best")
    for candidate in DWA_SETTINGS:
        add("validation","val_dwa_"+candidate["name"],"validation",episodes=nval,method="dwa",K=1,dwa_setting=candidate)
    # Repair ALL historical policy comparisons; other legacy mechanisms are checked nominally.
    for condition,env in CONDITIONS.items():
        for method in ("ppo","fsrl"):
            add("repair","repair_"+condition+"_"+method,"legacy",env,method=method,K=1,mapping="legacy")
    for method in ("gate","mlp","independent","rule","apf","cbf","cbf_qp"):
        add("repair","repair_nominal_"+method,"legacy",method=method,K=1 if method in ("apf","cbf","cbf_qp") else 30,mapping="legacy")
    add("repair","repair_fsrl_best","legacy",method="fsrl",K=1,checkpoint="best",mapping="legacy")
    # Main independent test bank. Native update regimes remain explicitly recorded.
    for condition,env in CONDITIONS.items():
        for method in ("ppo","fsrl","dwa","apf","cbf","cbf_qp","mlp"):
            add("native","native_"+condition+"_"+method,env=env,method=method,K=30 if method=="mlp" else 1,
                **({"dwa_setting":"validation_selected"} if method=="dwa" else {}))
    for method in ("gate","rule","independent"):
        add("native","native_nominal_"+method,method=method,K=30)
    for condition in ("nominal","corridor","utrap","dynamic"):
        add("checkpoint","checkpoint_"+condition+"_fsrl_best",env=CONDITIONS[condition],method="fsrl",K=1,checkpoint="best")
    # Gate x shared-rules x hold interval x mapping, with fixed exported MLP weights.
    for mapping in ("legacy","physical"):
        for k in (10,30):
            for gate in (False,True):
                for rules in (False,True):
                    add("ablation","ablation_"+mapping+str(k)+"_G"+str(int(gate))+"R"+str(int(rules)),
                        method="mlp",mapping=mapping,K=k,gate=gate,rules=rules)
    for gate,rules in ((False,True),(True,False),(False,False)):
        add("ablation","gate_scaffold_G"+str(int(gate))+"R"+str(int(rules)),method="gate",K=30,gate=gate,rules=rules,mapping="legacy")
    # Matched decision-update opportunities, no per-step shared modulation; PPO remains K=1 backbone.
    for method in ("mlp","apf","rule","cbf","cbf_qp","dwa"):
        for k in (1,10,30):
            add("cadence","cadence_"+method+"_"+str(k),method=method,K=k,gate=False,rules=False,**({"dwa_setting":"validation_selected"} if method=="dwa" else {}))
    # Isolate the two command-mapping mechanisms, not just "old vs new".
    for method in ("rule","mlp","apf"):
        for condition in ("nominal","corridor","utrap"):
            for mapping in ("legacy","scale_only","stop_only","physical"):
                add("mapping","mapping_"+method+"_"+condition+"_"+mapping,env=CONDITIONS[condition],method=method,K=30,mapping=mapping)
    # Sensitivity is predeclared, not test-bank threshold selection.
    for very in (-.3,-.2,-.1):
        for close in (-.1,0.,.1):
            if very<close:
                add("sensitivity","sensitivity_"+str(very)+"_"+str(close),method="mlp",K=30,
                    very_close_threshold=very,close_threshold=close)
    for progress in (.005,.02):
        add("sensitivity","progress_"+str(progress),method="mlp",K=30,progress_threshold_m=progress)
    for k in (10,30):
        for fault in ("zero","empirical","iid_fail","tail20","tail161","burst3","burst5"):
            for fallback in ("hold","emergency"):
                add("virtual","virtual_"+str(k)+"_"+fault+"_"+fallback,method="mlp",K=k,execution="virtual",fault=fault,fallback=fallback)
    for fault in ("empirical","tail20","burst3"):
        for fallback in ("hold","emergency"):
            add("virtual","virtual_legacy_"+fault+"_"+fallback,method="mlp",K=30,execution="virtual",fault=fault,fallback=fallback,mapping="legacy")
    for fault in ("tail161","burst5"):
        add("virtual","virtual_unguarded_"+fault,method="mlp",K=30,execution="virtual",fault=fault,
            request_deadline_s=None,response_ttl_s=None,decision_ttl_s=None)
    # Body coverage and sampling assumptions: empirical geometry diagnostics, never a certificate.
    for condition in ("nominal","corridor","utrap"):
        for lookahead in (.04,.08,.12):
            add("geometry","body_cbf_"+condition+"_"+str(lookahead),
                env={**CONDITIONS[condition],"lidar_mode":"nearest_sampled","collision_mode":"swept_disk"},
                method="cbf_qp",K=1,lookahead_m=lookahead,cbf_radius_m=.17+lookahead)
        for method in ("ppo","mlp","dwa"):
            add("geometry","geometry_"+condition+"_"+method,
                env={**CONDITIONS[condition],"lidar_mode":"nearest_sampled","collision_mode":"swept_disk"},
                method=method,K=30 if method=="mlp" else 1,**({"dwa_setting":"validation_selected"} if method=="dwa" else {}))
    gaps={"ideal":{},"actuation03":{"actuation_tau_s":.3},"actuation06":{"actuation_tau_s":.6},
          "command_delay2":{"command_delay_steps":2},"localization":{"localization_drift_per_sqrt_m":.02},
          "scan_delay":{"scan_delay_steps":3},"scan_burst":{"scan_burst_steps":5},
          "combined":{"actuation_tau_s":.3,"command_delay_steps":2,"localization_drift_per_sqrt_m":.02,"lidar_noise_std":.05,"scan_burst_steps":5}}
    for gap,env in gaps.items():
        for method in ("ppo","mlp","dwa","cbf_qp","fsrl"):
            add("domain","domain_"+gap+"_"+method,
                env={"lidar_mode":"nearest_sampled","collision_mode":"swept_disk",**env},method=method,K=30 if method=="mlp" else 1,
                **({"dwa_setting":"validation_selected"} if method=="dwa" else {}))
    # Genuine network execution. Each full nominal row has 1000 trials; fault rows 500 paired trials.
    for k in (10,30):
        add("live","live_async_"+str(k),episodes=settings["live_episodes"],method="gemini",K=k,execution="wallclock")
        add("live","live_blocked_"+str(k),episodes=settings["live_episodes"],method="gemini",K=k,execution="blocked")
    for k in (10,30):
        for fault in ("tail20","tail161","burst3","burst5"):
            add("live","live_fault_"+str(k)+"_"+fault,episodes=settings["live_stress_episodes"],method="gemini",K=k,execution="wallclock",fault=fault)
        add("live","live_emergency_"+str(k),episodes=settings["live_stress_episodes"],method="gemini",K=k,execution="wallclock",fault="burst3",fallback="emergency")
    for gate,rules in ((False,True),(True,False),(False,False)):
        add("live","live_ablation_G"+str(int(gate))+"R"+str(int(rules)),episodes=settings["live_stress_episodes"],method="gemini",K=30,execution="wallclock",gate=gate,rules=rules)
    return jobs
