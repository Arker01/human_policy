"""Opt-in package overlay, active only in a shell sourcing a SCDP bundle."""
import os
from pathlib import Path

if os.environ.get('HAT_SCDP_ONLINE') == '1':
    bundle = Path(os.environ['HAT_BUNDLE']).resolve()
    if (bundle / 'scdp_adapter.pt').is_file():
        import hdt
        overlay = Path(__file__).resolve().parents[1] / 'hdt'
        hdt.__path__[:0] = [str(overlay), str(bundle / 'source/hdt')]
