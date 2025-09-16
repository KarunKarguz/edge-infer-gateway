from pathlib import Path
import runpy
import sys

# Ensure same behavior as benchmark.py when invoked directly
this_dir = Path(__file__).resolve().parent
sys.argv[0] = str(this_dir / "benchmark.py")
runpy.run_path(str(this_dir / "benchmark.py"), run_name="__main__")
