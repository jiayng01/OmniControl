import os
import re
import csv
from pathlib import Path


def parse_log_file(file_path):
    metrics = {}
    state = None
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Section headers
            if "========== R_precision Summary" in line:
                state = "rprec"
                continue
            if "========== FID Summary" in line:
                state = "fid"
                continue
            if "========== Diversity Summary" in line:
                state = "diversity"
                continue
            if "========== Trajectory Error Summary" in line:
                state = "traj"
                continue

            # Parse metrics based on state
            if state == "rprec" and line.startswith("---> [vald]"):
                m = re.search(r"\(top 3\) Mean:\s*([0-9.]+)", line)
                if m:
                    metrics["R-precision (Top 3)"] = float(m.group(1))
                state = None

            elif state == "fid" and line.startswith("---> [vald]"):
                m = re.search(r"Mean:\s*([0-9.]+)", line)
                if m:
                    metrics["FID"] = float(m.group(1))
                state = None

            elif state == "diversity" and line.startswith("---> [vald]"):
                m = re.search(r"Mean:\s*([0-9.]+)", line)
                if m:
                    metrics["Diversity"] = float(m.group(1))
                state = None

            elif state == "traj" and line.startswith("---> [vald]"):
                # Extract multiple trajectory metrics from one line
                m50 = re.search(r"\(traj_fail_50cm\): Mean:\s*([0-9.]+)", line)
                k50 = re.search(r"\(kps_fail_50cm\): Mean:\s*([0-9.]+)", line)
                km = re.search(r"\(kps_mean_err\(m\)\): Mean:\s*([0-9.]+)", line)
                if m50:
                    metrics["Traj. err (50 cm)"] = float(m50.group(1))
                if k50:
                    metrics["Loc. err (50 cm)"] = float(k50.group(1))
                if km:
                    metrics["Avg. err"] = float(km.group(1))
                state = None

    return metrics


def parse_time_file(time_file_path):
    aits = None
    with open(time_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if "Average inference time per sample:" in line:
                m = re.search(r"Average inference time per sample:\s*([0-9.]+)", line)
                if m:
                    aits = float(m.group(1))
    return aits


def extract_filename_info(filename):
    info = {}
    # Density
    d = re.search(r"density(\d+)", filename)
    if d:
        info["Density"] = int(d.group(1))

    # Method and Timesteps
    if "ddim" in filename:
        info["Method"] = "DDIM"
        t = re.search(r"ddim(\d+)", filename)
        if t:
            info["Timesteps"] = int(t.group(1))
    elif "dpm2" in filename or "dpm3" in filename:
        m = re.search(r"(dpm[23])", filename)
        if m:
            info["Method"] = m.group(1).upper()
        n = re.search(r"nfe(\d+)", filename)
        if n:
            info["Timesteps"] = int(n.group(1))
    else:
        # Default to DDPM if no DDIM or DPM tag found
        info["Method"] = "DDPM"
        # For DDPM, timesteps depend on filename suffix
        if filename.endswith("_1000.log"):
            info["Timesteps"] = 0
        else:
            info["Timesteps"] = 1000

    return info


def main(log_dir, output_csv):
    log_dir = Path(log_dir)
    rows = []
    for log_file in log_dir.rglob("*.log"):
        fname = log_file.name
        # Parse metrics from file
        metrics = parse_log_file(log_file)
        if not metrics:
            continue
        # Parse average inference time
        time_dir = (log_file.parent / ".." / "times").resolve()
        time_file = time_dir / (fname[:-4] + "_time.log")
        aits = None
        if time_file.exists():
            aits = parse_time_file(time_file)
        metrics["AITS"] = aits
        # Extract info from filename
        info = extract_filename_info(fname)
        # Combine info and metrics
        row = {**info, **metrics, "Filename": fname}
        rows.append(row)

    # CSV headers
    headers = [
        "Filename",
        "Method",
        "Timesteps",
        "Density",
        "R-precision (Top 3)",
        "FID",
        "Diversity",
        "Traj. err (50 cm)",
        "Loc. err (50 cm)",
        "Avg. err",
        "AITS",
    ]

    # Write to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    import sys

    main("save/omnicontrol_ckpt/actual", "res.csv")
