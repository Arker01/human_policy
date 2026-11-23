import sys,json,pathlib,h5py,numpy as np,torch
sys.path.insert(0,str(pathlib.Path(__file__).parent/'hdt'))
from auxiliary_tasks import *
root=pathlib.Path(__file__).resolve().parents[1]
# Test true 3D composition and gradients, including orientation sensitivity.
a=torch.zeros(2,4,128);a[...,83]=a[...,87]=a[...,33]=a[...,37]=1
a[...,13:28]=torch.randn(2,4,15)*.04;a[...,43:58]=torch.randn(2,4,15)*.04
p=a.clone().requires_grad_();zero,cm=geometry_loss(p,a,torch.zeros(2,128),torch.ones(2,128),torch.ones(2,4,2),torch.zeros(2,4,dtype=torch.bool));assert zero.item()==0
p=a.clone();p[...,30]+=.02;p.requires_grad_();loss,cm=geometry_loss(p,a,torch.zeros(2,128),torch.ones(2,128),torch.ones(2,4,2),torch.zeros(2,4,dtype=torch.bool));assert abs(cm.item()-1)<1e-4;loss.backward();assert p.grad[...,30].abs().sum()>0
from pytorch3d.transforms import rotation_6d_to_matrix
v=torch.randn(20,6);assert torch.allclose(rotation_6d_rows(v),rotation_6d_to_matrix(v))
head=ReadinessHead(32);src=torch.randn(3,32,4,5,requires_grad=True);pos=torch.randn_like(src)
out=head(src,pos);loss=balanced_readiness_loss(out,torch.randint(0,2,(3,4)).float(),torch.ones(3,4));loss.backward();assert src.grad.abs().sum()>0
# No supervision means no spurious gradient, finite loss.
assert balanced_readiness_loss(out.detach(),torch.zeros(3,4),torch.zeros(3,4)).item()==0
report={}
for split,entries in json.load(open(root/'configs/replace50.json')).items():
 counts=np.zeros((4,2),np.int64);evcounts=np.zeros(4,np.int64);episodes=0;examples=[]
 for e in entries:
  for name in e['file_list']:
   with h5py.File(pathlib.Path(e['dataset_path'])/name) as f:
    states=f['observation.state'][:];fps=float(f.attrs['fps'])
   target,mask,weights,events=transition_labels(states,fps)
   counts[:,0]+=(target*mask).sum(0).astype(int);counts[:,1]+=((1-target)*mask).sum(0).astype(int)
   for hand,kind,onset,end in events:
    col=hand*2+(kind=='open');evcounts[col]+=1
    assert target[onset:,col].sum()>=0
    # Each event-positive window ends strictly before onset.
    assert target[max(0,onset-int(.2*fps)):onset,col].sum()>0
   if events and len(examples)<8: examples.append([name,events])
   episodes+=1
 report[split]=dict(episodes=episodes,positive_negative_by_head=counts.tolist(),events_by_head=evcounts.tolist(),examples=examples)
print(json.dumps(report,indent=2))
(root/'results/label_counts.json').write_text(json.dumps(report,indent=2))
print('AUXILIARY TESTS PASSED')
