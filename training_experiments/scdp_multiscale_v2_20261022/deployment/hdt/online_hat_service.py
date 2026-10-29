"""Keep the existing service CLI and transport; substitute an isolated SCDP loader."""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('hdt._scdp_original_service', REPO / 'hdt/online_hat_service.py')
original = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = original
spec.loader.exec_module(original)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scdp_online import make_online_class

original.OnlineHATPolicy = make_online_class(original)
main = original.main

if __name__ == '__main__':
    main()
