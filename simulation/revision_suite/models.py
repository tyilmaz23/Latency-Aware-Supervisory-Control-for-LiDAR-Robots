import pickle,re
import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO
from baseline_eval_suite import sanitize_decision,nav_context_to_features,independent_template_decision
from .common import ROOT,read_json

class Policies:
    def __init__(self):
        torch.set_num_threads(1)
        self.ppo=PPO.load(ROOT/"assets/ppo/model.zip",device="cpu")
        self.actors={}
    def ppo_action(self,obs):
        return np.asarray(self.ppo.predict(obs,deterministic=True)[0],dtype=np.float32)
    def fsrl_action(self,obs,checkpoint="final"):
        if checkpoint not in self.actors:self.actors[checkpoint]=FSRLActor(ROOT/("assets/fsrl/"+checkpoint+"_actor.pt"))
        return self.actors[checkpoint](obs)

class FSRLActor:
    """Deterministic FSRL/Tianshou ActorProb mean: ReLU MLP then tanh(mu).
    Critics and optimizer are not needed for evaluation. Parameters retain exact checkpoint tensors.
    """
    def __init__(self,path):
        payload=torch.load(path,map_location="cpu",weights_only=True);state=payload.get("model",payload)
        prefix="actor.preprocess.model.model."
        keys=sorted([k for k in state if k.startswith(prefix) and k.endswith(".weight")],key=lambda x:int(x[len(prefix):].split(".")[0]))
        expected=read_json(ROOT/"assets/fsrl/config.json")["hidden_sizes"]
        assert [state[k].shape[0] for k in keys]==expected
        assert state[keys[0]].shape[1]==126
        layers=[]
        for key in keys+["actor.mu.model.0.weight"]:
            weight=state[key];bias=state[key.replace(".weight",".bias")]
            layer=nn.Linear(weight.shape[1],weight.shape[0])
            with torch.no_grad():layer.weight.copy_(weight);layer.bias.copy_(bias)
            layers.append(layer)
            if key.startswith(prefix):layers.append(nn.ReLU())
        self.net=nn.Sequential(*layers,nn.Tanh()).eval()
    def __call__(self,obs):
        with torch.inference_mode():return self.net(torch.as_tensor(np.asarray(obs),dtype=torch.float32).reshape(1,-1))[0].numpy()

class Predictor:
    def __init__(self,kind):
        self.kind=kind
        if kind in ("mlp","independent"):
            path=ROOT/("assets/"+("mlp" if kind=="mlp" else "independent")+"/model.pkl")
            with open(path,"rb") as f:self.payload=pickle.load(f)
    def predict(self,ctx):
        x=nav_context_to_features(ctx).reshape(1,-1)
        if self.kind=="mlp":
            values=self.payload["reg_model"].predict(x)[0]
            result=dict(mode=str(self.payload["mode_model"].predict(x)[0]),speed_scale=float(values[0]),turn_scale=float(values[1]),turn_bias=float(values[2]))
        elif self.kind=="independent":
            result=independent_template_decision(str(self.payload["model"].predict(x)[0]),ctx)
        else:raise ValueError(self.kind)
        return sanitize_decision(result,ctx)
