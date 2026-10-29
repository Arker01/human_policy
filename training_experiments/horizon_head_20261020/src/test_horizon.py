import sys,pathlib,torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent/'hdt'))
from auxiliary_tasks import HorizonHead,horizon_loss,horizon_label_key

def test_head():
 torch.manual_seed(0)
 head=HorizonHead(16);hs=torch.randn(4,100,16,requires_grad=True)
 logits=head(hs);assert logits.shape==(4,100)
 assert torch.all(torch.diff(logits,dim=-1)<=0)
 target=torch.zeros_like(logits);target[:,:30]=1
 valid=torch.ones_like(logits,dtype=torch.bool);valid[2:]=False;valid[0,70:]=False
 loss=horizon_loss(logits,target,valid);loss.backward()
 assert torch.isfinite(loss) and head.hazard.weight.grad.norm()>0
 assert hs.grad[2:].abs().sum()==0
 empty=horizon_loss(logits.detach(),target,torch.zeros_like(valid));assert empty==0
 # Human path names overlap across datasets: cache keys must not.
 assert horizon_label_key('/data/human_old/wholebody-155.hdf5')!=horizon_label_key('/data/human_new/wholebody-155.hdf5')
 print('Horizon monotonicity, non-saturated initialization, masking, empty robot batch, gradient and cache collision checks passed')
if __name__=='__main__':test_head()
