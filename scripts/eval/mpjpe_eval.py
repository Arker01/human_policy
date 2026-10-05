#!/usr/bin/env python3
"""data/eval_mpjpe_batch.py 的薄包装，只补一件事：把 yaml 里的 zero_state_dims 传进模型。

为什么需要：eval_mpjpe_batch.py 的 _make_act_policy() 从 yaml 里读 backbone/层数等，
但没有读 model.zero_state_dims。ACT 里这个掩码不是纯 buffer——DETRVAE.__init__ 只有
在拿到 zero_state_dims 时才会 register_buffer('state_keep_mask')，forward 里也只有
self.zero_state_dims is not None 才会乘掩码（detr_vae.py:538, 868）。所以不传的话，
nostate 那几个 run 在 eval 时会拿到训练时从没见过的完整 128 维状态，数字是错的。

不改仓库里的 eval 脚本，只在这里 monkeypatch ACTPolicy 的构造参数。
其余参数原样透传给 data/eval_mpjpe_batch.py 的 main()。
"""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "hdt"))   # hdt/policy.py 里是 `from detr.main import ...`

import yaml as _yaml  # noqa: E402
import hdt.policy as _hp  # noqa: E402

_ZERO = None
for i, a in enumerate(sys.argv):
    if a == "--policy-config-yaml":
        cfg = _yaml.safe_load(open(sys.argv[i + 1]))
        z = (cfg.get("model") or {}).get("zero_state_dims")
        if isinstance(z, str):
            lo, hi = z.split(":")
            _ZERO = (int(lo), int(hi))
        elif z is not None:
            _ZERO = tuple(int(v) for v in z)

if _ZERO is not None:
    _Orig = _hp.ACTPolicy

    class _Patched(_Orig):
        def __init__(self, args_override):
            args_override = dict(args_override)
            args_override["zero_state_dims"] = _ZERO
            super().__init__(args_override)

    _hp.ACTPolicy = _Patched
    print(f"[mpjpe_eval] zero_state_dims={_ZERO} 已注入", flush=True)

_spec = importlib.util.spec_from_file_location(
    "eval_mpjpe_batch", REPO / "data" / "eval_mpjpe_batch.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_mod.main()
