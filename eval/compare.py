"""Day 3: print a before/after table from saved results.   python -m eval.compare"""
import json
import pathlib

files = sorted(pathlib.Path("results").glob("*.json"))
if not files:
    raise SystemExit("No results yet - run eval.run_eval first.")
runs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
keys = sorted({k for r in runs for k in r["summary"]})
print(f"{'run':28s}" + "".join(f"{k[:18]:>20s}" for k in keys))
for r in runs:
    print(f"{r['label'][:27]:28s}" + "".join(f"{r['summary'].get(k, '-'):>20s}" for k in keys))
print("\nConfig per run:")
for r in runs:
    print(f"  {r['label']}: {r['config']}")
