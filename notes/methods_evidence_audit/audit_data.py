from pathlib import Path
import json, csv
import numpy as np
import zarr

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
TASKS={'ring_over_peg_redo':'boom_40in','cable_stringing':'boom_40in','whiteboard_checkmark':'whiteboard_checkmark_boom_40in'}
summary={}; split_export={}; rows=[]
for task,name in TASKS.items():
 result={}
 for kind,suffix in [('raw',''),('cleaned','_cleaned')]:
  path=ROOT/'data'/task/(name+suffix+'.zarr'); g=zarr.open_group(str(path),mode='r'); a=dict(g.attrs)
  ends=g['meta/episode_ends'][:]; starts=np.r_[0,ends[:-1]]; ts=g['data/timestamp_ns'][:]; state=g['data/state'][:]; d3=state[:,2]
  spans=np.array([(ts[e-1]-ts[s])/1e9 for s,e in zip(starts,ends)])
  ptps=np.array([np.ptp(d3[s:e]) for s,e in zip(starts,ends)])
  meta=g['meta']; wall=((meta['episode_stop_monotonic_ns'][:]-meta['episode_start_monotonic_ns'][:])/1e9)
  result[kind]={'path':str(path.relative_to(ROOT)),'episodes':len(ends),'steps':int(ends[-1]),'sample_span_seconds':float(spans.sum()),'metadata_recording_seconds':float(wall.sum()),'nominal_seconds_at_30hz':int(ends[-1])/30,'d3_min_m':float(np.nanmin(d3)),'d3_max_m':float(np.nanmax(d3)),'d3_finite':bool(np.isfinite(d3).all()),'d3_quantiles_m':np.nanquantile(d3,[0,.01,.5,.99,1]).tolist(),'within_episode_d3_range_min_m':float(ptps.min()),'within_episode_d3_range_median_m':float(np.median(ptps)),'within_episode_d3_range_max_m':float(ptps.max()),'episodes_with_d3_range_over_1mm':int((ptps>.001).sum()),'d3_equals_joint_command':bool(np.array_equal(d3,g['data/joint_position_command'][:,2],equal_nan=True)),'camera_products':sorted(set(s.get('device_product','unknown') for s in a.get('imu_sessions',{}).values())),'camera_device_ids':sorted(set(s.get('device_id','unknown') for s in a.get('imu_sessions',{}).values())),'git_revisions':a.get('git_revisions'),'state_fields':a.get('state_fields'),'attribute_keys':list(a)}
  if kind=='cleaned':
   clean=a.get('clean',{}); result['cleaning']={k:v for k,v in clean.items() if not isinstance(v,(list,dict))};result['cleaning_list_keys']={k:len(v) for k,v in clean.items() if isinstance(v,list)}
   if task=='ring_over_peg_redo': mapping=[m['source_episode'] for m in clean['output_episode_mapping']]
   elif task=='whiteboard_checkmark':mapping=clean['output_episode_to_source_episode']
   else:
    report=json.loads((ROOT/'data'/task/(name+'_cleaned_report.json')).read_text());mapping=[m['source_episode'] for m in report['episode_mapping']]
   n=len(ends); val=sorted(np.random.default_rng(0).permutation(n)[:round(.1*n)].tolist()); train=sorted(set(range(n))-set(val))
   configs=[]
   for p in (ROOT/'checkpoints').rglob('config.json'):
    c=json.loads(p.read_text())
    if '/'+task+'/' in c.get('dataset',''):
     configs.append(str(p.relative_to(ROOT)));assert c['train_episodes']==train and c['val_episodes']==val and c['val_fraction']==.1 and c['seed']==0
   split_export[task]={'indexing':'zero-based cleaned episode indices','saved_configs_verified':configs,'train_episodes':train,'val_episodes':val,'output_episode_to_source_episode':mapping,'source_trials_in_both_splits':sorted(set(mapping[i] for i in train)&set(mapping[i] for i in val))}
   split_export[task]['verification_scope']='Saved Full config verified; Angles split not independently verified (only checkpoint available).' if task!='whiteboard_checkmark' else 'Saved Full and Angles configs verified and identical.'
   result['split']={'train':len(train),'validation':len(val),'source_trials_in_both_splits':split_export[task]['source_trials_in_both_splits']}
   for i,(s,e) in enumerate(zip(starts,ends)):
    rows.append({'task':task,'cleaned_episode':i,'source_episode':mapping[i],'split':'validation' if i in val else 'train','steps':int(e-s),'sample_span_seconds':float(spans[i]),'metadata_recording_seconds':float(wall[i]),'d3_min_m':float(np.min(d3[s:e])),'d3_max_m':float(np.max(d3[s:e])),'d3_range_m':float(ptps[i])})
 summary[task]=result
(OUT/'dataset_summary.json').write_text(json.dumps(summary,indent=2)+'\n');(OUT/'episode_splits.json').write_text(json.dumps(split_export,indent=2)+'\n')
with (OUT/'episodes.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for task,r in summary.items():
 print(task,json.dumps({k:v for k,v in r.items() if k not in ['cleaning_list_keys']}))
