"""
Launch a SageMaker training job for rooftop segmentation.

SageMaker downloads the train channel to /opt/ml/input/data/train. Your S3 data must be:
  s3://<bucket>/<prefix>/images/
  s3://<bucket>/<prefix>/masks/
  s3://<bucket>/<prefix>/filenames/
    train_filenames_roof_outline_v1.txt
    val_filenames_roof_outline_v1.txt
    test_filenames_roof_outline_v1.txt

So use a single prefix (e.g. data/) containing images/, masks/, filenames/.

Prerequisites:
  pip install sagemaker
  AWS CLI configured (or env AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
  IAM role with SageMaker execution and S3 access (or pass --role).

If you see SSLError during source upload to S3: try again (script retries 3x), use another
network, or disable VPN/proxy/antivirus that might intercept HTTPS. Running from EC2/Cloud9
in the same region as the bucket often avoids local SSL issues.
"""

import argparse
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import boto3
from sagemaker.pytorch import PyTorch
from sagemaker.session import Session

# Only these files are uploaded to SageMaker (keeps tarball small, reduces SSL upload failures)
SAGEMAKER_SOURCE_FILES = ("train.py", "dataset.py", "requirements.txt")


def _make_minimal_source_dir(repo_root: Path) -> str:
    """Build a minimal source dir with only train.py, dataset.py, requirements.txt.
    Reduces upload size and multipart count, which helps avoid SSLError on flaky connections.
    """
    repo_root = Path(repo_root)
    tmp = Path(tempfile.mkdtemp(prefix="sagemaker_src_"))
    try:
        for name in SAGEMAKER_SOURCE_FILES:
            src = repo_root / name
            if src.exists():
                shutil.copy2(src, tmp / name)
            else:
                raise FileNotFoundError(f"Required for SageMaker upload: {src}")
        return str(tmp)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def get_args():
    p = argparse.ArgumentParser(description="Launch SageMaker rooftop segmentation training")
    p.add_argument(
        "--role",
        type=str,
        default=os.environ.get("SAGEMAKER_EXECUTION_ROLE"),
        help="SageMaker execution IAM role ARN (or set SAGEMAKER_EXECUTION_ROLE)",
    )
    p.add_argument(
        "--region",
        type=str,
        default=os.environ.get("AWS_REGION", "us-east-2"),
        help="AWS region for SageMaker and S3 (must match your bucket; default: us-east-2)",
    )
    p.add_argument(
        "--train-data",
        type=str,
        default="s3://01262026tannereischen/data/",
        help="S3 URI to data prefix containing images/, masks/, filenames/",
    )
    p.add_argument(
        "--output-path",
        type=str,
        default="s3://01262026tannereischen/output",
        help="S3 URI for model/output artifacts",
    )
    p.add_argument(
        "--job-name",
        type=str,
        default=None,
        help="SageMaker job name (default: roof-seg-<timestamp>)",
    )
    p.add_argument(
        "--instance-type",
        type=str,
        default="ml.m5.xlarge",
        help="SageMaker instance type (default: ml.m5.xlarge; use ml.g4dn.xlarge for GPU if you have quota)",
    )
    p.add_argument(
        "--instance-count",
        type=int,
        default=1,
        help="Number of instances",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=30,
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=256,
    )
    p.add_argument(
        "--encoder",
        type=str,
        default="resnet34",
    )
    p.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Path to training source (default: directory containing this script)",
    )
    return p.parse_args()


def main():
    args = get_args()
    if not args.role:
        raise SystemExit(
            "Missing --role. Set SAGEMAKER_EXECUTION_ROLE or pass --role <arn>"
        )

    repo_root = Path(__file__).resolve().parent
    minimal_dir = None
    if args.source_dir:
        source_dir = args.source_dir
    else:
        # Use minimal bundle (train.py, dataset.py, requirements.txt only) to keep upload small
        source_dir = _make_minimal_source_dir(repo_root)
        minimal_dir = source_dir
    train_data = args.train_data.rstrip("/")

    hyperparameters = {
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "lr": args.lr,
        "image-size": args.image_size,
        "encoder": args.encoder,
    }

    job_name = args.job_name or f"roof-seg-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

    boto_session = boto3.Session(region_name=args.region)
    sagemaker_session = Session(boto_session=boto_session)

    estimator = PyTorch(
        entry_point="train.py",
        source_dir=source_dir,
        role=args.role,
        framework_version="2.0",
        py_version="py310",
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        hyperparameters=hyperparameters,
        output_path=args.output_path,
        job_name=job_name,
        sagemaker_session=sagemaker_session,
    )

    # Channel name "train" -> container sees /opt/ml/input/data/train (SM_CHANNEL_TRAIN)
    # Retry fit() in case of transient SSL/network errors (e.g. SSLError during S3 multipart upload)
    last_err = None
    try:
        for attempt in range(3):
            try:
                estimator.fit({"train": train_data})
                print("Training job finished. Model artifacts in", args.output_path)
                return
            except Exception as e:
                last_err = e
                if "SSL" in str(type(e).__name__) or "SSL" in str(e):
                    wait = (attempt + 1) * 30
                    print(f"Upload/SSL error (attempt {attempt + 1}/3): {e}. Retrying in {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    raise
        raise last_err
    finally:
        if minimal_dir and os.path.isdir(minimal_dir):
            shutil.rmtree(minimal_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
