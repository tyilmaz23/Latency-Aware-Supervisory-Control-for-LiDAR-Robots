from collections import deque
import math
import numpy as np
from baseline_eval_suite import EvaluationEnv, DT, distance_m_to_normalized_lidar
from .common import scene,digest

def point_rect_distance(point,rectangles):
    r=np.asarray(rectangles,dtype=float)
    if len(r)==0:return float("inf")
    d=np.maximum(np.maximum(r[:,:2]-np.asarray(point),np.asarray(point)-r[:,2:]),0)
    return float(np.sqrt((d*d).sum(axis=1)).min())
def swept_rect_distance(start,end,rectangles):
    """Exact distance between a line segment and axis-aligned rectangles."""
    r=np.asarray(rectangles,dtype=float)
    if len(r)==0:return float("inf")
    a=np.asarray(start,dtype=float);b=np.asarray(end,dtype=float);v=b-a
    entering=np.zeros(len(r));leaving=np.ones(len(r));possible=np.ones(len(r),dtype=bool)
    for axis in range(2):
        if abs(v[axis])<1e-14:
            possible&=(a[axis]>=r[:,axis])&(a[axis]<=r[:,axis+2])
        else:
            t1=(r[:,axis]-a[axis])/v[axis];t2=(r[:,axis+2]-a[axis])/v[axis]
            entering=np.maximum(entering,np.minimum(t1,t2));leaving=np.minimum(leaving,np.maximum(t1,t2))
    if np.any(possible&(entering<=leaving)):return 0.
    corners=np.stack([r[:,:2],r[:,[0,3]],r[:,[2,1]],r[:,2:]],axis=1).reshape(-1,2)
    vv=float(v@v)
    t=np.clip(((corners-a)@v)/max(vv,1e-30),0,1)
    corner_dist=np.sqrt(((corners-(a+t[:,None]*v))**2).sum(axis=1)).min()
    return min(float(corner_dist),point_rect_distance(a,r),point_rect_distance(b,r))

class RevisionEnv(EvaluationEnv):
    def __init__(self,config=None):
        self.cfg=dict(config or {});self.noise_rng=np.random.default_rng(0)
        self.pose_error=np.zeros(3);self.estimated_distance_buffer=deque(maxlen=5);self.scan_buffer=deque();self.last_scan=None
        self.command_buffer=deque();self.command_buffer_w=deque();self.actual_command=np.array([-1.,0.])
        self.step_motion=0.;self.true_clearance=1.;self.true_collision=False
        self.geometric=self.cfg.get("collision_mode","legacy_rays")=="swept_disk"
        super().__init__(env_variant=self.cfg.get("env_variant","baseline"),
            lidar_noise_std=float(self.cfg.get("lidar_noise_std",0)),beam_dropout_rate=float(self.cfg.get("beam_dropout_rate",0)))
        if self.geometric:self.COLLISION_DIST=0. # Ground-truth disk test replaces ray threshold.
    def lidar_readings(self):
        """Vectorized original 100-point sampler; wall-order semantics are selectable."""
        walls=np.asarray(self.walls+self.dynamic_walls,dtype=float)
        if len(walls)==0:return np.ones(len(self.LIDAR_ANGLES))
        distances=np.linspace(.01,1.,100)
        angles=self.agent_pos[2]+self.LIDAR_ANGLES
        px=self.agent_pos[0]+np.cos(angles)[:,None]*distances
        py=self.agent_pos[1]+np.sin(angles)[:,None]*distances
        hit=(px[:,:,None]>=walls[None,None,:,0])&(px[:,:,None]<=walls[None,None,:,2])&(py[:,:,None]>=walls[None,None,:,1])&(py[:,:,None]<=walls[None,None,:,3])
        if self.cfg.get("lidar_mode","legacy_first_wall")=="nearest_sampled":
            chosen=hit.any(axis=2);found=chosen.any(axis=1)
        else:
            per_wall=hit.any(axis=1);found=per_wall.any(axis=1);first=per_wall.argmax(axis=1)
            chosen=hit[np.arange(len(angles)),:,first]
        measured=np.where(found,distances[chosen.argmax(axis=1)],1.)
        return np.clip(measured,.16,1.)
    def _apply_observation_corruption(self,obs):
        out=np.asarray(obs,dtype=np.float32).copy()
        ranges=.16+.42*(np.clip(out[6:],-1,1)+1)
        if self.lidar_noise_std:ranges+=self.noise_rng.normal(0,self.lidar_noise_std,ranges.shape)
        if self.beam_dropout_rate:
            ranges[self.noise_rng.random(len(ranges))<self.beam_dropout_rate]=1.
        burst=int(self.cfg.get("scan_burst_steps",0))
        if burst and self.step_count%40<burst and self.last_scan is not None:ranges=self.last_scan.copy()
        else:self.last_scan=ranges.copy()
        lag=int(self.cfg.get("scan_delay_steps",0))
        self.scan_buffer.append(ranges.copy())
        if lag:
            while len(self.scan_buffer)>lag+1:self.scan_buffer.popleft()
            ranges=self.scan_buffer[0]
        else:self.scan_buffer.clear()
        out[6:]=distance_m_to_normalized_lidar(ranges)
        std=float(self.cfg.get("localization_drift_per_sqrt_m",0))
        if std:
            self.pose_error+=self.noise_rng.normal(0,std*math.sqrt(max(self.step_motion,.001)),3)
            estimated=self.agent_pos.astype(float)+self.pose_error
            d=float(np.linalg.norm(self.target_pos-estimated[:2]))
            heading=self.calculate_theta_to_target(estimated,self.target_pos)
            out[0:2]=[math.sin(heading),math.cos(heading)];out[4]=d/100.
            if not self.estimated_distance_buffer:self.estimated_distance_buffer.extend([d]*5)
            else:self.estimated_distance_buffer.append(d)
            out[5]=(self.estimated_distance_buffer[-1]-self.estimated_distance_buffer[0])/100.
        return out
    def reset(self,seed=None,options=None):
        seed=int(seed if seed is not None else 0)
        self.rng=np.random.default_rng(seed)
        self.noise_rng=np.random.default_rng(np.random.SeedSequence([seed,773341]))
        self.pose_error=np.zeros(3);self.estimated_distance_buffer.clear();self.scan_buffer.clear();self.last_scan=None;self.command_buffer.clear();self.command_buffer_w.clear()
        self.actual_command=np.array([-1.,0.]);self.step_motion=0.;self.true_collision=False
        obs,info=super().reset()
        fixture=(options or {}).get("fixture",self.cfg.get("fixture"))
        if fixture:
            self.agent_pos=np.asarray(fixture["start_pose"],dtype=np.float32);self.target_pos=np.asarray(fixture["goal_xy"],dtype=np.float32)
            self.walls=[tuple(r) for r in fixture["walls_xyxy"]];self.dynamic_walls=[];self._moving_obstacles=[];self._static_variant_walls=[]
            if np.max(np.abs(self.agent_pos[:2]))>4.8 or np.max(np.abs(self.target_pos))>4.8:raise ValueError("Measured fixture outside original 10m workspace")
            distance=float(np.linalg.norm(self.agent_pos[:2]-self.target_pos));self.distance_buffer=deque([distance]*5,maxlen=5)
            self.path=[self.agent_pos[:2].copy()];self.all_targets[-1]=self.target_pos.copy();self.distance_to_target_start=distance-.5
            self.scan_buffer.clear();self.last_scan=None;self.pose_error=np.zeros(3);self.estimated_distance_buffer.clear()
            self.noise_rng=np.random.default_rng(np.random.SeedSequence([seed,773341]))
            theta=self.calculate_theta_to_target(self.agent_pos,self.target_pos)
            obs=np.r_[[math.sin(theta),math.cos(theta),0,0,distance/100,0],distance_m_to_normalized_lidar(self.lidar_readings())].astype(np.float32)
            obs=self._apply_observation_corruption(obs)
        self.scene=scene(self);self.scene_hash=digest(self.scene)
        self.true_clearance=point_rect_distance(self.agent_pos[:2],self.walls+self.dynamic_walls)
        return obs,info
    def step(self,action):
        command=np.clip(np.asarray(action,dtype=float),-1,1)
        delay=int(self.cfg.get("command_delay_steps",0));delay_w=int(self.cfg.get("command_delay_w_steps",delay))
        commanded=command.copy()
        self.command_buffer.append(float(commanded[0]));self.command_buffer_w.append(float(commanded[1]))
        command[0]=self.command_buffer.popleft() if len(self.command_buffer)>delay else -1.
        command[1]=self.command_buffer_w.popleft() if len(self.command_buffer_w)>delay_w else 0.
        tau=float(self.cfg.get("actuation_tau_s",0));tau_w=float(self.cfg.get("actuation_tau_w_s",tau))
        physical=np.array([.11*(command[0]+1)*self.cfg.get("actuation_gain_v",1),2.84*command[1]*self.cfg.get("actuation_gain_w",1)])
        old=np.array([.11*(self.actual_command[0]+1),2.84*self.actual_command[1]])
        factors=np.array([1-math.exp(-DT/tau) if tau else 1.,1-math.exp(-DT/tau_w) if tau_w else 1.])
        physical=old+factors*(physical-old)
        command=np.clip([physical[0]/.11-1,physical[1]/2.84],-1,1)
        self.actual_command=command.copy()
        old_position=self.agent_pos[:2].copy();old_obstacles=list(self.dynamic_walls)
        obs,reward,terminated,truncated,info=super().step(command)
        self.step_motion=float(np.linalg.norm(self.agent_pos[:2]-old_position))
        radius=float(self.cfg.get("body_radius_m",.17))
        # Moving rectangles evaluated at both endpoints; this is not a continuous moving-body certificate.
        clearance=swept_rect_distance(old_position,self.agent_pos[:2],self.walls+old_obstacles)
        clearance=min(clearance,swept_rect_distance(old_position,self.agent_pos[:2],self.walls+self.dynamic_walls))
        self.true_clearance=clearance;self.true_collision=clearance<radius
        goal=float(np.linalg.norm(self.agent_pos[:2]-self.target_pos))<self.GOAL_REACHED_DIST
        info.update(ground_truth_disk_collision=bool(self.true_collision),true_clearance_m=float(clearance),
                    goal_condition=bool(goal),legacy_reason=info.get("reason",""))
        if self.geometric:
            collision=self.true_collision
            if collision:
                terminated=True;info["reason"]="collision";self.target_outcomes[-1]="collision"
            elif goal:info["reason"]="target_reached"
            elif truncated:info["reason"]="max_steps_reached"
            self.step_count-=1
            try:reward=self._reward(float(np.linalg.norm(self.agent_pos[:2]-self.target_pos)),goal,collision,
                                   .11*(command[0]+1),2.84*command[1],float(np.min(self.lidar_readings())),
                                   self.distance_buffer[-1]-self.distance_buffer[0])
            finally:self.step_count+=1
        return obs,float(reward),bool(terminated),bool(truncated),info
