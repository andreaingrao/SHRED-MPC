#!/usr/bin/env python3
"""Verify curated hashes, normalization, force convention and headline metrics."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from shred_mpc.observables import lift_time_series_p

ROOT=Path(__file__).resolve().parents[1]


def main():
    data=ROOT/'data/curated'
    manifest=json.loads((data/'provenance.json').read_text())
    for row in manifest['files']:
        p=data/row['path']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError(f'Curated hash mismatch: {p}')
    with (data/'candidate_statistics.csv').open() as f:rows=list(csv.DictReader(f))
    eligible=[r for r in rows if int(r['eligible_for_statistics'])]
    selected=[r for r in eligible if int(r['shred_selected'])]
    assert (len(rows),len(eligible),len(selected))==(250,210,42)
    assert sum(int(r['choice_mistake']) for r in selected)==27
    assert len({(r['simulation_id'],r['simulation_cycle']) for r in selected})==42
    metrics=json.loads((data/'metrics.json').read_text())
    for col,key in [('pressure_relative_l2_error','pressure_relative_l2_quartiles'),('lift_relative_l2_error','lift_relative_l2_quartiles')]:
        np.testing.assert_allclose(np.quantile([float(r[col]) for r in eligible],[.25,.5,.75]),metrics[key],rtol=1e-12)
    points=np.load(data/'points.npy')
    with (data/'representative_raw_cfd_frame0.csv').open() as f:raw=list(csv.DictReader(f))
    xy=np.array([[float(r['Points:0']),float(r['Points:1'])] for r in raw]);np.testing.assert_array_equal(points,xy)
    p=np.array([float(r.get('p_raw',r['p'])) for r in raw]);left=xy[:,0]<=xy[:,0].min()+max(1e-9,np.ptp(xy[:,0])*1e-6)
    row=json.loads((data/'representative_row.json').read_text())
    with np.load(data/'representative_prediction.npz') as a:
        np.testing.assert_array_equal(((p-p[left].mean())/.125).astype(np.float32),a['pressure_cfd_reference'][0])
        for field,lift in [('pressure_prediction','lift_prediction'),('pressure_cfd_reference','lift_cfd_reference')]:
            np.testing.assert_allclose(lift_time_series_p(a[field],points),a[lift],atol=1e-7,rtol=1e-7)
        ratio=np.linalg.norm(a['pressure_prediction']-a['pressure_cfd_reference'])/np.linalg.norm(a['pressure_cfd_reference'])
        np.testing.assert_allclose(ratio,float(row['pressure_relative_l2_error']),rtol=1e-6)
    with np.load(data/'selected_tracking.npz') as t:
        assert len(t['controller_sample'])==1250
        np.testing.assert_array_equal(t['controller_sample'],np.arange(51,1301))
    print('Verified 13 curated file hashes; 250/210 rows; 42 decisions; real pressure normalization, lift and result summaries.')

if __name__=='__main__':main()
