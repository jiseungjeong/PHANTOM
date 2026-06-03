#!/usr/bin/env python3
"""Zoomed visualization of suspect peak files (read-only).

Files flagged by phase diagnostics as 'weakest<0.30 (possible fake peak)' or
'gapCV>0.5 (irregular)' are drawn large, to eyeball whether they are truly fake
peaks vs genuinely weak/irregular motions.

Each subplot = fused envelope + 5 peaks (red lines) + crop bands. Weak peaks get
an arrow marker.
Output: reports/suspect_peaks.png
"""
from __future__ import annotations
import sys, types
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

THIS=Path(__file__).resolve(); REPO=THIS.parents[2]
DATA=REPO/"Data"; OUT=THIS.parent/"reports"; OUT.mkdir(exist_ok=True)
L=REPO/"analysis"/"jiho_juho_shim_active_window_classifier_minimal"
sys.path.insert(0,str(L))
import rule_base_classifier as rbc
sys.modules["analysis"]=types.ModuleType("analysis")
sys.modules["analysis.active_window_classifier_minimal"]=types.ModuleType("analysis.active_window_classifier_minimal")
sys.modules["analysis.active_window_classifier_minimal.rule_base_classifier"]=rbc
import build_single_repetition_dataset as bsr
sys.modules["analysis.active_window_classifier_minimal.build_single_repetition_dataset"]=bsr
import build_peak_repetition_dataset as bp
sys.path.insert(0, str(THIS.parent))
import phase1_segment as p1     # use the improved choose_real_peaks

WIN=["#7fb3ff","#84d68a","#ffd166","#c792ea","#ff8fa3"]

# Suspect files flagged by diagnostics (specified by filename)
SUSPECTS=[
    "JUHO/JUHO_Lead Uppercut_20260517_183223.csv",
    "JUHO/JUHO_Rear Uppercut_20260517_183714.csv",
    "SHIM/SHIM_Rear Uppercut_20260530_154413.csv",
    "JUHO/JUHO_Rear Jab_20260517_181950.csv",
    "JUHO/JUHO_Lead Hook_20260517_182400.csv",
    "JUHO/JUHO_Lead Jab_20260517_181825.csv",
    "SHIM/SHIM_Lead Jab_20260530_154018.csv",
    "SHIM/SHIM_Rear Jab_20260530_154112.csv",
    "JUHO/JUHO_Rear Hook_20260517_182943.csv",
    "JIHO/JIHO_Left Roll_20260517_194311.csv",
    "JIHO/JIHO_Rear Jab_20260517_192943.csv",
    "SHIM/SHIM_Right Slip_20260530_154712.csv",
]

def resolve(rel):
    subj=rel.split("/")[0].lower(); name=rel.split("/")[1]
    return DATA/f"{subj}_data"/name

def main():
    n=len(SUSPECTS); ncols=3; nrows=(n+ncols-1)//ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(6*ncols,2.6*nrows),squeeze=False)
    for k,rel in enumerate(SUSPECTS):
        ax=axes[k//ncols][k%ncols]; path=resolve(rel)
        if not path.exists():
            ax.text(.5,.5,"missing: "+path.name,ha="center",fontsize=7); continue
        data=rbc.read_csv_sources(path); _,meta=rbc.extract_features(path)
        jump=float(meta.get("jump_cut_s",0))
        times,env,dur,_=bp.motion_envelope(data,jump)
        peaks,pm=p1.choose_real_peaks(times,env,dur)     # improved selection
        windows=bp.windows_from_peaks(times,env,peaks,dur) if len(peaks)>=p1.MIN_REPS else []
        windows=p1.uniform_windows(windows,dur)          # improvement 5: uniform width (generous)
        ax.plot(times,env,color="#1f4e79",lw=1.3); ymax=max(env.max(),1e-6)
        for i,w in enumerate(windows):
            ax.add_patch(Rectangle((w["start_s"],0),w["end_s"]-w["start_s"],ymax,
                                   color=WIN[i%5],alpha=0.4,zorder=0))
            pt=w["peak_time_s"]; pv=w["peak_score"]
            ax.axvline(pt,color="#d72638",lw=1.0)
            if pv < 0.30*ymax:                      # highlight weak peak
                ax.annotate("weak?",(pt,pv),fontsize=8,color="red",weight="bold")
        ax.set_yticks([])
        lab=path.name.split('_',1)[1].rsplit('_',2)[0]
        subj=path.parent.name.replace('_data','').upper()
        if len(peaks)>=p1.MIN_REPS:
            ttl=f"{subj} | {lab}  {len(peaks)}pk g{pm.get('gap_cv')} h{pm.get('height_cv')}"; col="black"
        else:
            ttl=f"{subj} | {lab}  INVALID({pm.get('status')})"; col="red"
        ax.set_title(ttl, fontsize=9, color=col)
    for k in range(n,nrows*ncols): axes[k//ncols][k%ncols].axis("off")
    fig.suptitle("Suspect peak files (fused envelope + 5 peaks + crop bands; 'weak?'=peak<30% of max)",fontsize=12,y=0.997)
    fig.tight_layout(rect=[0,0,1,0.98])
    out=OUT/"suspect_peaks.png"; fig.savefig(out,dpi=120)
    print(f"saved -> {out}  ({n} suspect files)")

if __name__=="__main__": main()
