from pathlib import Path
import json
import numpy as np
import torch
R=Path(__file__).resolve().parents[2]; O=Path(__file__).resolve().parent
paths=[R/'checkpoints'/s/'policy_epoch_0500.pt' for s in ['ring_over_peg_redo/nautilus-boom-40in-baseline','ring_over_peg_redo/nautilus-boom-40in-joint-angles','src-cable-stringing-boom-40in-baseline','nautilus-cable-stringing-boom-40in-joint-angles','src-whiteboard-boom-40in-baseline','src-whiteboard-boom-40in-joint-angles']]
cs={}
for p in paths:
 d=torch.load(p,map_location='cpu',weights_only=False);c=d['config']; c.setdefault('state_input','full');c.setdefault('imu_input','none')
 result={'policy':c,'normalizer':d['normalizer'],'model_shapes':{k:list(v.shape) for k,v in d['model'].items()},'optimizer_groups':[{k:v for k,v in g.items() if k!='params'} for g in d['optimizer']['param_groups']]}
 mp=p.parent/'metrics.jsonl'
 if mp.exists():
  metrics=[json.loads(x) for x in mp.read_text().splitlines()];result['last_metrics']=metrics[-1];print('METRIC KEYS',p.parent.name,metrics[-1]); vals=[m for m in metrics if 'val_action_mse' in m]; result['best_validation']=min(vals,key=lambda m:m['val_action_mse']) if vals else None
 cs[str(p.relative_to(R))]=result
 print('CHECKPOINT',p.parent.name,c,'normalizer d3 bounds',d['normalizer']['state_low'][2],d['normalizer']['state_high'][2])
(O/'checkpoint_summary.json').write_text(json.dumps(cs,indent=2)+'\n')
summary={}
for root in sorted((R/'deployment_runs').iterdir()):
 if not any(s in root.name for s in ['ring_over_peg','cable_stringing','whiteboard']) or 'ROLLOUTS' in root.name:continue
 cp=root/'config.json'
 if not cp.exists():continue
 config=json.loads(cp.read_text());eps=[]
 for p in sorted(root.rglob('episode.json')):
  if p.parent.name.startswith(('X_','2m_test')):continue
  m=json.loads(p.read_text());values=[m['initial_joints'][2],m['final_joints'][2]]; controls=0;outside=0;grasp_true=0
  for line in (p.parent/m['data']).open():
   d=json.loads(line)
   if d.get('event')=='control':
    values.append(d['joint_position'][2]);controls+=1;grasp_true+=bool(d.get('action',[0]*7)[6] >= 0.5)
  eps.append({'path':str(p.relative_to(R)),'initial_d3':values[0],'min_d3':min(values),'max_d3':max(values),'control_samples':controls,'grasp_true_samples':grasp_true})
 summary[root.name]={'config':config,'n_inventoried_episodes':len(eps),'initial_d3_range':[min(e['initial_d3'] for e in eps),max(e['initial_d3'] for e in eps)] if eps else None,'d3_range':[min(e['min_d3'] for e in eps),max(e['max_d3'] for e in eps)] if eps else None,'episodes':eps}
 print('ROLLOUT',root.name,len(eps),summary[root.name]['initial_d3_range'],summary[root.name]['d3_range'])
(O/'rollout_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
