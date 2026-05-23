"""
ENERGIVANU — Battery Signal Engine
SAFE / PREPARE / CRITICAL → Tesla Megapack commands
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from src.config import Config


@dataclass
class Action:
    ts: datetime; sig: int; name: str; peak_mw: float; ttp: int
    soc: float; dis_mw: float; thr: float; conf: float; reason: str


class SignalEngine:
    N = {0:"SAFE",1:"PREPARE",2:"CRITICAL"}
    def __init__(self, cfg: Config):
        self.s = cfg.signal; self.b = cfg.battery; self.hist=[]

    def generate(self, pred, soc, mc=0.8, ts=None):
        ts = ts or datetime.now()
        peak=float(np.max(pred)); avg=float(np.mean(pred))
        ttp=int(np.argmax(pred))*5

        self.hist.append(peak)
        if len(self.hist)>100: self.hist=self.hist[-100:]
        stab = max(0,1-np.std(self.hist[-10:])/20) if len(self.hist)>5 else 0.5
        conf = 0.5*mc + 0.5*stab

        avail = soc/100*self.b.capacity_mwh
        md = min(self.b.max_discharge_mw, avail/0.25)

        if peak>=self.s.critical_mw:
            df=peak-self.s.critical_mw; d=min(df*1.2,md)
            t=min(50,max(0,(df-d)/max(peak,1)*100))
            r=f"CRITICAL: {peak:.1f}MW in {ttp}s. Discharge {d:.1f}MW. SOC:{soc:.0f}%."
            sig=2
        elif peak>=self.s.warning_mw:
            d,t=0,0
            r=f"WARNING: {peak:.1f}MW in {ttp}s. Pre-charge. SOC:{soc:.0f}%."
            sig=1
        else:
            d,t=0,0
            r=f"SAFE: {avg:.1f}MW avg. Peak {peak:.1f}MW. SOC:{soc:.0f}%."
            sig=0

        return Action(ts,sig,self.N[sig],peak,ttp,soc,d,t,conf,r)
