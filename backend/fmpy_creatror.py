#!/usr/bin/env python3
"""
Quick FMU → CSV dataset generator (FMPy)

Usage examples:
  python fmpy_run.py --fmu MyModel.fmu --start 0 --stop 60 --step 0.1 --out dataset.csv
  python fmpy_run.py --fmu MyModel.fmu --vars y x1 --start 0 --stop 10 --step 0.05
  python fmpy_run.py --fmu MyModel.fmu --inputs inputs.csv --start 0 --stop 20 --step 0.1
  python fmpy_run.py --fmu MyModel.fmu --init-json '{"k": 2.5, "x_start": 0.0}'
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from fmpy import read_model_description, simulate_fmu


def list_variables(fmu_path):
    md = read_model_description(fmu_path)
    vars_info = []
    for sv in md.modelVariables:
        vars_info.append({
            "name": sv.name,
            "causality": getattr(sv, "causality", None),
            "variability": getattr(sv, "variability", None),
            "type": sv.type.__class__.__name__.replace("Type","").lower()
        })
    return pd.DataFrame(vars_info)


def build_input_struct(inputs_csv: Path):
    """CSV must have column 'time' plus one column per input variable."""
    df = pd.read_csv(inputs_csv)
    if 'time' not in df.columns:
        raise ValueError("inputs CSV must contain a 'time' column")
    cols = [c for c in df.columns if c != 'time']
    dtype = [('time', np.float64)] + [(c, np.float64) for c in cols]
    arr = np.zeros(len(df), dtype=dtype)
    arr['time'] = df['time'].to_numpy(dtype=np.float64)
    for c in cols:
        arr[c] = df[c].to_numpy(dtype=np.float64)
    return arr


def auto_outputs(fmu_path):
    """Pick a reasonable default set of outputs if user didn’t specify."""
    md = read_model_description(fmu_path)
    outs = []
    # Prefer declared outputs
    for sv in md.modelVariables:
        if getattr(sv, "causality", "") == "output":
            outs.append(sv.name)
    if outs:
        return outs
    # Fallback: any reals that aren’t parameters
    for sv in md.modelVariables:
        if getattr(sv, "variability", "") not in ("tunable", "fixed"):
            outs.append(sv.name)
    return outs


def main():
    ap = argparse.ArgumentParser(description="FMU → CSV with FMPy")
    ap.add_argument("--fmu", required=True, type=Path, help="Path to .fmu")
    ap.add_argument("--start", type=float, default=0.0, help="Start time")
    ap.add_argument("--stop", type=float, required=True, help="Stop time")
    ap.add_argument("--step", type=float, default=None, help="Output interval (sampling step)")
    ap.add_argument("--vars", nargs="*", default=None, help="Variables to record (default: auto)")
    ap.add_argument("--inputs", type=Path, default=None, help="CSV with time + input signals")
    ap.add_argument("--init-json", type=str, default=None, help="JSON dict of start values")
    ap.add_argument("--out", type=Path, default=Path("dataset.csv"), help="Output CSV path")
    ap.add_argument("--fmi-type", choices=["CoSimulation", "ModelExchange"], default=None,
                    help="Force FMI type (auto if omitted)")
    args = ap.parse_args()

    fmu_path = args.fmu
    if not fmu_path.exists():
        raise FileNotFoundError(f"FMU not found: {fmu_path}")

    # Decide outputs
    outputs = args.vars if (args.vars and len(args.vars) > 0) else auto_outputs(fmu_path)

    # Optional inputs
    input_struct = build_input_struct(args.inputs) if args.inputs else None

    # Optional start values
    start_values = json.loads(args.init_json) if args.init_json else None

    # Prepare simulate_fmu kwargs (avoid passing None to keep compatibility)
    sim_kwargs = dict(filename=str(fmu_path),
                      start_time=args.start,
                      stop_time=args.stop)
    if args.fmi_type:
        sim_kwargs["fmi_type"] = args.fmi_type
    if outputs:
        sim_kwargs["output"] = outputs
    if input_struct is not None:
        sim_kwargs["input"] = input_struct
    if start_values:
        sim_kwargs["start_values"] = start_values
    if args.step:
        # FMPy uses `output_interval` as the uniform sampling interval
        sim_kwargs["output_interval"] = args.step

    print("▶ Running simulation with settings:")
    print(json.dumps({
        "fmu": str(fmu_path),
        "start_time": args.start,
        "stop_time": args.stop,
        "output_interval": args.step,
        "outputs": outputs,
        "fmi_type": args.fmi_type or "auto",
        "inputs_csv": str(args.inputs) if args.inputs else None
    }, indent=2))

    # Run
    result = simulate_fmu(**sim_kwargs)

    # Convert to DataFrame & save
    df = pd.DataFrame(result)
    # Ensure 'time' is first
    cols = list(df.columns)
    if 'time' in cols:
        cols.remove('time')
        cols = ['time'] + cols
    df = df[cols]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"✅ Wrote {len(df)} rows × {len(df.columns)} columns → {args.out}")

    # Also dump a quick variables listing for convenience
    try:
        lv = list_variables(fmu_path)
        with open(args.out.with_suffix(".vars.txt"), "w", encoding="utf-8") as f:
            f.write(lv.to_string(index=False))
        print(f"ℹ️  Saved variable list → {args.out.with_suffix('.vars.txt')}")
    except Exception as e:
        print(f"(Skipping variable list: {e})")


if __name__ == "__main__":
    main()
