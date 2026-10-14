#!/usr/bin/env python3
"""Extract small, real evidence samples without modifying source archives."""
from pathlib import Path
import argparse, csv, hashlib, json, shutil
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit('Output must be absent or empty; refusing to overwrite evidence.')
    output.mkdir(parents=True, exist_ok=True)
    run = source / 'shred_stat_results/run_20260713_121325_067474'
    rows = list(csv.DictReader((run/'candidate_statistics.csv').open()))
    eligible = [r for r in rows if int(r['eligible_for_statistics']) == 1]
    selected = sorted([r for r in rows if int(r['shred_selected']) == 1], key=lambda r: int(r['simulation_cycle']))
    eligible_selected = [r for r in selected if int(r['eligible_for_statistics']) == 1]
    median = np.median([float(r['pressure_relative_l2_error']) for r in eligible_selected])
    rep = min(eligible_selected, key=lambda r: (abs(float(r['pressure_relative_l2_error'])-median), int(r['simulation_cycle'])))
    manifest = {'experiment': run.name, 'models': sorted({r['run_id'] for r in rows}), 'selection': 'Representative is selected candidate nearest median pressure relative L2 error among eligible SHRED-selected candidates; ties resolved by earliest cycle.', 'sources': [], 'transforms': {}, 'limitations': ['No trained checkpoint found.', 'Current configuration differs from archived 12-pressure-sensor model.', 'Tracking combines selected CFD candidate first25 branches; direct main-case lift trace was not archived. Prefix correspondence was checked against later recorded sensor histories.', 'Use controller sample index: current CFD templates use steadyState/SIMPLE; this evidence does not establish physical transient accuracy.']}
    def record(path):
        manifest['sources'].append({'path': str(path.relative_to(source)), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    def copy(path, name):
        record(path);shutil.copyfile(path,output/name)
    copy(run/'candidate_statistics.csv','candidate_statistics.csv')
    copy(run/'run_metadata.json','run_metadata.json')
    copy(run/'training_events.csv','training_events.csv')
    copy(run/'lift_target_sequence_simulation_000.csv','lift_target_sequence.csv')
    copy(run/'points_vec.npy','points.npy')
    rep_path = source/rep['shred_prediction_archive']
    copy(rep_path,'representative_prediction.npz')
    copy(source/rep['candidate_cfd_directory']/'metadata.json','representative_metadata.json')
    copy(source/rep['candidate_cfd_directory']/'CFD_0.csv','representative_raw_cfd_frame0.csv')
    (output/'representative_row.json').write_text(json.dumps(rep,indent=2)+'\n')
    points=np.load(run/'points_vec.npy')
    csv_rows=list(csv.DictReader((output/'representative_raw_cfd_frame0.csv').open()))
    csv_points=np.array([[float(x['Points:0']),float(x['Points:1'])] for x in csv_rows])
    assert np.array_equal(points,csv_points)
    raw=np.array([float(x.get('p_raw',x['p'])) for x in csv_rows]);left=csv_points[:,0]<=csv_points[:,0].min()+np.ptp(csv_points[:,0])*1e-6
    normalized=((raw-raw[left].mean())/.125).astype(np.float32)
    with np.load(rep_path) as arc: assert np.array_equal(normalized,arc['pressure_cfd_reference'][0])
    lookup={int(r['simulation_cycle']):r for r in selected}
    historical_ids=None;checks=[]
    for c in [10,20,30,39,40]:
        with np.load(source/lookup[c]['shred_prediction_archive']) as arc: sensors=arc['sensor_input'][:,:-1]
        refs=[]
        for prev in range(c-4,c):
            with np.load(source/lookup[prev]['shred_prediction_archive']) as arc: refs.append(arc['pressure_cfd_reference'][:25])
        ref=np.concatenate(refs);mse=((ref[:,:,None]-sensors[:,None,:])**2).mean(0);ids=mse.argmin(0);rms=np.sqrt(mse[ids,np.arange(sensors.shape[1])]);assert np.all(rms==0)
        if historical_ids is not None: assert np.array_equal(historical_ids,ids)
        historical_ids=ids;checks.append({'cycle':c,'values_compared':int(sensors.size),'maximum_rms':float(rms.max())})
    (output/'historical_sensors.json').write_text(json.dumps({'sensor_ids':historical_ids.tolist(),'coordinates':points[historical_ids].tolist(),'inference':'Exact float32 equality of each sensor_input pressure column to preceding four selected candidate CFD first25 branches, unique minimum-MSE point columns.', 'checks':checks, 'current_code_sensor_ids':[837,2178,2279,3144,3214,3351,3455]},indent=2)+'\n')
    trace={k:[] for k in ['controller_sample','cycle','lift_prediction','lift_cfd_selected_branch','omega_applied','lift_target','statistics_eligible','cfd_file_time']}
    checks_controls=[]
    for row in selected:
        path=source/row['shred_prediction_archive'];meta_path=source/row['candidate_cfd_directory']/'metadata.json';record(path);record(meta_path)
        meta=json.loads(meta_path.read_text());c=int(row['simulation_cycle']);t=int(row['controller_timestep']);h=int(meta['control_horizon'])
        with np.load(path) as arc:
            values={'controller_sample':t+np.arange(1,h+1),'cycle':np.full(h,c),'lift_prediction':arc['lift_prediction'][:h],'lift_cfd_selected_branch':arc['lift_cfd_reference'][:h],'omega_applied':arc['omega_future'][:h],'lift_target':np.full(h,float(row['lift_target'])),'statistics_eligible':np.full(h,int(row['eligible_for_statistics'])),'cfd_file_time':float(meta['base_cfd_time'])+.2*np.arange(1,h+1)}
            for k,v in values.items():trace[k].append(v)
            if c+1 in lookup:
                with np.load(source/lookup[c+1]['shred_prediction_archive']) as nxt:
                    assert np.array_equal(arc['omega_future'][:h],nxt['sensor_input'][25:50,-1])
                    assert np.array_equal(arc['pressure_cfd_reference'][:h,historical_ids],nxt['sensor_input'][-h:,:-1])
                checks_controls.append(c)
    trace={k:np.concatenate(v) for k,v in trace.items()};np.savez_compressed(output/'selected_tracking.npz',**trace)
    with (output/'selected_tracking.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(trace);writer.writerows(zip(*(trace[k] for k in trace)))
    manifest['transforms']['selected_tracking']='For each simulation0 cycle0..49, select shred_selected==1 and concatenate lift_prediction, lift_cfd_reference and omega_future indices0:25. Label samples controller_timestep+1..+25; target is candidate row lift_target. CFD file time uses archived base_cfd_time plus 0.2*(1..25); not evidence of physical unsteady time.'
    manifest['transforms']['tracking_checks']={'checked_cycles':checks_controls,'control_check':'candidate first25 omega equals next input control-history rows25:50 exactly','pressure_check':'candidate first25 pressure at recovered12 sensor IDs equals next sensor_input final25 pressure rows exactly','unverified_last_branch':'cycle49 has no next archived input to validate replay correspondence'}
    manifest['transforms']['point_order']='points.npy equals representative raw CFD coordinate columns exactly; normalized raw pressure using leftmost-x mean and q=.125 equals stored reference frame0 exactly.'
    quantiles=lambda k:[float(x) for x in np.quantile([float(r[k]) for r in eligible],[.25,.5,.75])]
    vals={'rows':len(rows),'eligible_rows':len(eligible),'cycles':len(selected),'eligible_cycles':len(eligible_selected),'models':len({r['run_id'] for r in rows}),'simulations':len({r['simulation_id'] for r in rows}),'pressure_relative_l2_quartiles':quantiles('pressure_relative_l2_error'),'lift_relative_l2_quartiles':quantiles('lift_relative_l2_error'),'wrong_choice_cycles':sum(int(r['choice_mistake']) for r in eligible_selected),'regret_quartiles':[float(x) for x in np.quantile([float(r['selected_cfd_regret']) for r in eligible_selected],[.25,.5,.75])],'regret_max':max(float(r['selected_cfd_regret']) for r in eligible_selected),'variability_unit':'Correlated candidate evaluations along one closed-loop trajectory from one model; quartiles are descriptive, not confidence intervals.','relative_l2_definition':'norm(prediction-reference.ravel()) / max(norm(reference.ravel()),1e-12), across 50 steps and all points for pressure; across 50 steps for lift.'}
    (output/'metrics.json').write_text(json.dumps(vals,indent=2)+'\n')
    manifest['files']=[{'path':str(p.relative_to(output)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()]
    (output/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output':str(output),'files':len(manifest['files'])+1,'bytes':sum(p.stat().st_size for p in output.iterdir()),'representative_cycle':rep['simulation_cycle'],'verified_following_histories':len(checks_controls)},indent=2))

if __name__=='__main__':main()
