"""
ENERGIVANU — Workload + Battery Co-Optimizer
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict
from enum import Enum
from src.config import Config


class Pri(Enum):
    CRIT=1; HIGH=2; MED=3; LOW=4

@dataclass
class Job:
    id: str; pri: Pri; pw: float; dur: float; dl: datetime
    throttle: bool=True; fmax: float=1800

@dataclass
class Result:
    sched: List[Dict]; batt: List[Dict]; dvfs: Dict
    stress: float; savings: float


class CoOptimizer:
    def __init__(self, cfg: Config):
        self.b = cfg.battery; self.g = cfg.grid

    def optimize(self, jobs, pred, soc, now):
        sc, bp, dv = [],[],[]
        sorted_j = sorted(jobs, key=lambda j: j.pri.value)
        hr = self.g.max_import_mw - float(np.max(pred))
        ab = min(self.b.max_discharge_mw, soc/100*self.b.capacity_mwh/0.25)
        sv = 0.0

        for j in sorted_j:
            if hr>=j.pw:
                sc.append({"job":j.id,"act":"RUN","freq":j.fmax})
                hr-=j.pw
            elif hr+ab>=j.pw:
                d=j.pw-hr; ab-=d
                sc.append({"job":j.id,"act":"RUN+BATT","freq":j.fmax})
                bp.append({"dis":d}); hr=0
            elif j.throttle and j.pri.value>=3:
                r=max(0.5,(hr/max(j.pw,1))**(1/3))
                nf=j.fmax*r; ap=j.pw*r**3
                sv+=j.pw-ap; dv[j.id]=nf
                sc.append({"job":j.id,"act":"THROTTLED","freq":nf,"save":(1-r**3)*100})
                hr-=ap
            elif j.pri.value>=3:
                dl=min((j.dl-now).total_seconds()/3600,24)
                sc.append({"job":j.id,"act":"DELAY","h":dl})
            else:
                sc.append({"job":j.id,"act":"EMERGENCY"})

        st=max(0,min(1,(float(np.max(pred))-self.g.max_import_mw*0.8)/(self.g.max_import_mw*0.2)))
        tp=sum(j.pw for j in jobs)
        return Result(sc,bp,dv,st,sv/max(tp,1)*100)
