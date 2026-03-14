"""
Download a trained roof segmentation model.

Supports two sources:
1. Hugging Face Hub (recommended): Downloads best.pt and history.json
2. SageMaker S3: Downloads from training job output

Usage:
  # From Hugging Face (default)
  python download_model.py --source huggingface --repo-id TannerEischen/roof-segmentation

  # From SageMaker S3
  python download_model.py --source sagemaker --job-name pytorch-training-2026-03-10-14-16-27-042
"""

import argparse
import os
import subprocess
import sys
import tarfile
from pathlib import Path


def get_args():
    p = argparse.ArgumentParser(description="Download roof segmentation model")
    p.add_argument(
        "--source",
        type=str,
        choices=["huggingface", "sagemaker"],
        default="huggingface",
        help="Download source: huggingface or sagemaker (default: huggingface)",
    )
    p.add_argument(
        "--repo-id",
        type=str,
        default="TannerEischen/roof-segmentation",
        help="Hugging Face repo ID (default: TannerEischen/roof-segmentation)",
    )
    p.add_argument(
        "--job-name",
        type=str,
        help="SageMaker training job name (required if source=sagemaker)",
    )
    p.add_argument(
        "--output-path",
        type=str,
        default="s3://01262026tannereischen/output",
        help="S3 URI prefix for SageMaker output",
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


def download_from_huggingface(repo_id: str, save_dir: Path):
    """Download model from Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    save_dir.mkdir(parents=True, exist_ok=True)

    files_to_download = ["best.pt", "history.json"]
    for filename in files_to_download:
        print(f"Downloading {filename} from {repo_id} ...")
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(save_dir),
            )
            print(f"Saved to {local_path}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}", file=sys.stderr)

    print(f"Done. Model files saved to {save_dir}/")


def download_from_sagemaker(job_name: str, output_path: str, save_dir: Path, region: str):
    """Download model from SageMaker S3 output."""
    output_path = output_path.rstrip("/")
    s3_uri = f"{output_path}/{job_name}/output/model.tar.gz"
    save_dir.mkdir(parents=True, exist_ok=True)
    local_tar = save_dir / "model.tar.gz"

    print(f"Downloading {s3_uri} ...")
    try:
        subprocess.run(
            ["aws", "s3", "cp", s3_uri, str(local_tar), "--region", region],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Download failed. Is the job complete? Check S3 or use a different --job-name.", file=sys.stderr)
        sys.exit(e.returncode)

    out_dir = save_dir / job_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting to {out_dir} ...")
    with tarfile.open(local_tar, "r:gz") as tf:
        tf.extractall(out_dir)
    local_tar.unlink()
    print(f"Done. Model files in {out_dir} (e.g. best.pt, history.json).")


def main():
    args = get_args()
    args.save_dir = Path(args.save_dir)

    if args.source == "huggingface":
        download_from_huggingface(args.repo_id, args.save_dir)
    elif args.source == "sagemaker":
        if not args.job_name:
            print("Error: --job-name is required when using sagemaker source", file=sys.stderr)
            sys.exit(1)
        download_from_sagemaker(args.job_name, args.output_path, args.save_dir, args.region)


if __name__ == "__main__":
    main()
