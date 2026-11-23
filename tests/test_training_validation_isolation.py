"""Validation at step zero must not replace the pending training minibatch."""
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'hdt'))
import hdt.main as training


class Progress:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, count):
        pass


def test_validation_keeps_training_batch_separate(monkeypatch, tmp_path):
    policy = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
    calls = []

    def forward(batch, model, training_step=0):
        calls.append((model.training, int(batch.item())))
        return {'loss': (model(batch.reshape(1, 1)) ** 2).mean()}

    class Accelerator:
        is_main_process = True

        def prepare(self, *items):
            return items

        def backward(self, loss):
            loss.backward()

    monkeypatch.setattr(training, 'forward_pass', forward)
    monkeypatch.setattr(training.accelerate.state, 'AcceleratorState', lambda: SimpleNamespace(process_index=0))
    monkeypatch.setattr(training, 'tqdm', Progress)
    monkeypatch.setattr(training, '_append_metrics_csv', lambda *a, **kw: None)
    monkeypatch.setattr(training, '_plot_metrics', lambda *a, **kw: None)
    monkeypatch.setattr(training, 'wandb', SimpleNamespace(run=None))
    monkeypatch.setattr(training, 'maybe_load_ckpt', lambda *a: (0, None))
    config = {'num_epochs': 2, 'ckpt_dir': str(tmp_path), 'seed': 0,
              'load_pretrained_path': None, 'val_and_jit_trace': False}
    training.train_fn(
        Accelerator(), [torch.tensor([1.]), torch.tensor([2.])],
        [torch.tensor([101.]), torch.tensor([102.])], policy, optimizer, config,
    )
    assert [marker for is_train, marker in calls if not is_train] == [101, 102]
    assert [marker for is_train, marker in calls if is_train] == [1, 2]
    assert (tmp_path / 'policy_last.ckpt').is_file()
