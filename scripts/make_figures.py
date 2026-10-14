#!/usr/bin/env python3
"""Regenerate figures from the curated July archive. No training or CFD.

Run from any working directory. Existing output directories are refused.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
BLUE='#245c83'; ORANGE='#bf6436'; TEAL='#167b76'; INK='#20323f'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,'axes.titleweight':'medium',
    'axes.edgecolor':'#aebac1','grid.color':'#dce3e7','grid.alpha':.7,'svg.fonttype':'none',
    'figure.facecolor':'white','savefig.facecolor':'white','svg.hashsalt':'shred-mpc'})


def save(fig,out,name):
    fig.savefig(out/f'{name}.svg',bbox_inches='tight',metadata={'Date':None})
    fig.savefig(out/f'{name}.png',dpi=160,bbox_inches='tight')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/curated')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--animation',action='store_true',help='Requires ffmpeg; renders a 5-second archive animation')
    args=parser.parse_args(); data=args.data; out=args.output
    out.mkdir(parents=True,exist_ok=False)
    pts=np.load(data/'points.npy'); a=np.load(data/'representative_prediction.npz')
    track=np.load(data/'selected_tracking.npz'); sensors=json.loads((data/'historical_sensors.json').read_text())
    with (data/'candidate_statistics.csv').open() as f: rows=list(csv.DictReader(f))
    eligible=[r for r in rows if int(r['eligible_for_statistics'])]
    chosen=[r for r in eligible if int(r['shred_selected'])]
    cp=np.array([float(r['pressure_relative_l2_error']) for r in eligible])
    lift=np.array([float(r['lift_relative_l2_error']) for r in eligible])
    summary={'eligible_candidate_rows':len(eligible),'eligible_decisions':len(chosen),
      'pressure_median_relative_l2':float(np.median(cp)), 'lift_median_relative_l2':float(np.median(lift)),
      'pressure_iqr':np.quantile(cp,[.25,.75]).tolist(),'lift_iqr':np.quantile(lift,[.25,.75]).tolist(),
      'choice_disagreements':sum(int(r['choice_mistake']) for r in chosen),
      'regret_median':float(np.median([float(r['selected_cfd_regret']) for r in chosen]))}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Geometry uses actual archived cell centres; sensor recovery is documented.
    fig,ax=plt.subplots(figsize=(7.4,5.2),layout='constrained')
    ax.scatter(pts[:,0],pts[:,1],s=2,color='#c5d1d8',rasterized=True,label='Archived cell centres')
    old=np.array(sensors['sensor_ids']);new=np.array(sensors['current_code_sensor_ids'])
    ax.scatter(pts[old,0],pts[old,1],s=55,facecolors='white',edgecolors=ORANGE,linewidths=1.6,label='12 historical pressure channels')
    ax.scatter(pts[new,0],pts[new,1],s=18,color=TEAL,label='7 current configured channels')
    ax.add_patch(Circle((0,0),2.5,color=INK));ax.text(0,0,'Rotating\ncylinder',color='white',ha='center',va='center',fontsize=9)
    ax.annotate('Inlet: 0.5 m/s',xy=(-9,1),xytext=(-9,4),fontsize=9)
    ax.arrow(-9,0,3,0,width=.04,head_width=.45,color=BLUE,length_includes_head=True)
    ax.set(xlim=(-10,10),ylim=(-10,10),xlabel='x [m]',ylabel='y [m]',title='Geometry and pressure observations')
    ax.set_aspect('equal');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),frameon=False,ncol=1,fontsize=9)
    save(fig,out,'geometry')
    # Method diagram, a code-drawn schematic, not simulation evidence.
    fig,ax=plt.subplots(figsize=(11,3.4));ax.axis('off');ax.set(xlim=(0,11),ylim=(0,3.4))
    boxes=[(.1,1.6,'Observe','Pressure history\n+ proposed rotation'),(2.9,1.6,'Predict','LSTM → shallow decoder\n→ pressure field'),(5.7,1.6,'Evaluate','Pressure-ring lift\n→ candidate cost'),(8.5,1.6,'Apply','Choose rotation\napply first 25 samples')]
    for x,y,title,body in boxes:
        ax.add_patch(Rectangle((x,y),2.4,1.35,facecolor='#f0f5f7',edgecolor='#c9d7df',linewidth=1))
        ax.text(x+.18,y+1.03,title,fontsize=13,weight='bold');ax.text(x+.18,y+.5,body,fontsize=10,va='center')
    for x in [2.5,5.3,8.1]:ax.annotate('',xy=(x+.4,2.27),xytext=(x,2.27),arrowprops={'arrowstyle':'->','color':TEAL,'lw':1.7})
    ax.annotate('',xy=(1.3,.75),xytext=(9.7,.75),arrowprops={'arrowstyle':'->','color':TEAL,'lw':1.7})
    ax.plot([9.7,9.7],[1.6,.75],color=TEAL);ax.plot([1.3,1.3],[.75,1.6],color=TEAL)
    ax.text(5.5,.23,'CFD feedback · update observations · repeat at the next control decision',ha='center',fontsize=10,color=TEAL)
    save(fig,out,'pipeline')
    pred=a['pressure_prediction'];ref=a['pressure_cfd_reference'];err=abs(pred-ref)
    lo=min(pred.min(),ref.min());hi=max(pred.max(),ref.max());emax=err.max()
    def fields(index,title):
        fig,axs=plt.subplots(1,3,figsize=(10.8,3.8),layout='constrained')
        for ax,values,name,cmap,vmin,vmax in zip(axs,[ref[index],pred[index],err[index]],['CFD reference','SHRED prediction','Absolute error'],['RdBu_r','RdBu_r','magma'],[lo,lo,0],[hi,hi,emax]):
            scatter=ax.scatter(pts[:,0],pts[:,1],c=values,cmap=cmap,vmin=vmin,vmax=vmax,s=7,marker='s',edgecolor='none',rasterized=True)
            ax.add_patch(Circle((0,0),2.5,facecolor='white',edgecolor=INK,lw=.6))
            ax.set(xlim=(-10,10),ylim=(-10,10),xlabel='x [m]',title=name);ax.set_aspect('equal')
            fig.colorbar(scatter,ax=ax,orientation='horizontal',shrink=.88,pad=.13,label='Cp' if name!='Absolute error' else '|Cp prediction − reference|')
        axs[0].set_ylabel('y [m]');fig.suptitle(title,fontsize=11)
        return fig
    save(fields(24,'Cycle 39 · selected candidate 0 · forecast position 25 / 50'),out,'fields')
    fig,axs=plt.subplots(2,1,figsize=(10.2,5.3),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[2.5,1]})
    x=track['controller_sample']
    axs[0].plot(x,track['lift_cfd_selected_branch'],color=BLUE,lw=1.2,label='Selected CFD branch')
    axs[0].plot(x,track['lift_prediction'],color=ORANGE,lw=.9,alpha=.85,label='Selected SHRED prediction')
    axs[0].step(x,track['lift_target'],where='post',color=INK,ls='--',lw=1.2,label='Target')
    axs[0].axvspan(x.min(),250,color='#f1f3f4',zorder=-1);axs[0].set_ylabel('Historical pressure-lift observable')
    axs[0].legend(loc='upper center',bbox_to_anchor=(.5,1.16),ncol=3,frameon=False,fontsize=9)
    axs[1].plot(x,track['omega_applied'],color=TEAL,lw=1.3);axs[1].set(xlabel='Controller sample index (not validated physical time)',ylabel='Rotation ω [rad/s]')
    for ax in axs:ax.grid(axis='y',alpha=.5)
    save(fig,out,'tracking')
    fig,axs=plt.subplots(1,2,figsize=(10.2,3.8),layout='constrained')
    for values,name,color in [(cp*100,'Pressure',BLUE),(lift*100,'Pressure lift',ORANGE)]:
        sorted_=np.sort(values);axs[0].plot(sorted_,np.arange(1,len(values)+1)/len(values),label=name,color=color,lw=2)
    axs[0].set(xlabel='Window relative L2 error [%]',ylabel='Fraction of eligible candidate rows',title='Field accuracy and force accuracy');axs[0].legend(frameon=False)
    cycles=[int(r['simulation_cycle']) for r in chosen];regret=[float(r['selected_cfd_regret']) for r in chosen]
    axs[1].stem(cycles,regret,linefmt=TEAL,markerfmt='o',basefmt=' ')
    axs[1].set(xlabel='Decision cycle',ylabel='CFD cost regret',title=f"{summary['choice_disagreements']} / {len(chosen)} choices differ from CFD minimum")
    axs[1].text(.02,.96,'One model · one trajectory',transform=axs[1].transAxes,va='top',fontsize=9,color='#637784')
    save(fig,out,'decision-quality')
    fig,axs=plt.subplots(2,1,figsize=(8.3,4.8),sharex=True,layout='constrained')
    h=np.arange(1,51);rnorm=np.linalg.norm(ref,axis=1)
    e=np.divide(np.linalg.norm(pred-ref,axis=1),rnorm,out=np.full(50,np.nan),where=rnorm>1e-12)
    axs[0].plot(h,e*100,color=BLUE,lw=2);axs[0].set(ylabel='Pressure relative L2 [%]',title='One selected forecast block · cycle 39 / candidate 0')
    axs[1].plot(h,a['lift_cfd_reference'],color=BLUE,label='CFD');axs[1].plot(h,a['lift_prediction'],color=ORANGE,label='SHRED')
    axs[1].axhline(0,color=INK,ls='--',label='Target');axs[1].set(xlabel='Forecast position (fixed trained offset; sample index)',ylabel='Pressure-lift observable')
    axs[1].legend(frameon=False,ncol=3)
    for ax in axs:
        ax.axvspan(.5,25.5,color=TEAL,alpha=.07);ax.axvline(25.5,color='#9eafb8',ls=':');ax.grid(axis='y',alpha=.4)
    axs[0].text(3,axs[0].get_ylim()[1]*.94,'Rotation ramp',fontsize=9,va='top');axs[0].text(29,axs[0].get_ylim()[1]*.94,'Hold',fontsize=9,va='top')
    save(fig,out,'forecast-block')
    if args.animation:
        if not shutil.which('ffmpeg'):raise RuntimeError('ffmpeg unavailable; static figures were generated')
        with tempfile.TemporaryDirectory(prefix='shred-frames-') as tmp:
            for k in range(50):
                fig=fields(k,f'Archived candidate forecast · position {k+1:02d}/50 · ω = {a["omega_future"][k]:.4f} rad/s')
                fig.savefig(Path(tmp)/f'{k:03d}.png',dpi=90);plt.close(fig)
            subprocess.run(['ffmpeg','-v','error','-framerate','10','-i',str(Path(tmp)/'%03d.png'),'-c:v','libx264','-crf','25','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'forecast.mp4')],check=True)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
