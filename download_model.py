"""
Download a trained model from SageMaker training job output (S3).

SageMaker uploads the contents of /opt/ml/model as model.tar.gz to:
  s3://<output-path>/<job-name>/output/model.tar.gz

After download, the script extracts the archive so you get best.pt, history.json, etc.
"""

import argparse
import os
import subprocess
import sys
import tarfile
from pathlib import Path


def get_args():
    p = argparse.ArgumentParser(description="Download SageMaker training model from S3")
    p.add_argument(
        "--job-name",
        type=str,
        required=True,
        help="SageMaker training job name (e.g. pytorch-training-2026-03-10-14-16-27-042)",
    )
    p.add_argument(
        "--output-path",
        type=str,
        default="s3://01262026tannereischen/output",
        help="S3 URI prefix where the job wrote its output (same as launcher --output-path)",
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        default=Path("models"),
        help="Local directory to save the model (default: models/)",
    )
    p.add_argument(
        "--region",
        type=str,
        default=os.environ.get("AWS_REGION", "us-east-2"),
        help="AWS region for S3",
    )
    return p.parse_args()


def main():
    args = get_args()
    output_path = args.output_path.rstrip("/")
    s3_uri = f"{output_path}/{args.job_name}/output/model.tar.gz"
    args.save_dir = Path(args.save_dir)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    local_tar = args.save_dir / "model.tar.gz"

    print(f"Downloading {s3_uri} ...")
    try:
        subprocess.run(
            ["aws", "s3", "cp", s3_uri, str(local_tar), "--region", args.region],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Download failed. Is the job complete? Check S3 or use a different --job-name.", file=sys.stderr)
        sys.exit(e.returncode)

    out_dir = args.save_dir / args.job_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting to {out_dir} ...")
    with tarfile.open(local_tar, "r:gz") as tf:
        tf.extractall(out_dir)
    local_tar.unlink()
    print(f"Done. Model files in {out_dir} (e.g. best.pt, history.json).")


if __name__ == "__main__":
    main()
