"""Native Linux Docker CLI/HTTP gate for an explicitly identified local image.

Uses only own UUID-named containers. No daemon startup, global cleanup, browser,
weight download or Docker-socket mount. The passed image must already exist.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def docker(*args):
    result = subprocess.run(["docker", *args], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Docker {args[0]} failed: {result.stderr[-2000:]}")
    return result.stdout.strip()


def inspect(container):
    return json.loads(docker("inspect", container))[0]


def request(base, route, content=None):
    headers = {}
    if content is not None:
        boundary = "roof-" + uuid.uuid4().hex
        body = (
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="fixture.png"\r\n'
                "Content-Type: image/png\r\n\r\n"
            ).encode()
            + content
            + f"\r\n--{boundary}--\r\n".encode()
        )
        headers["Content-Type"] = "multipart/form-data; boundary=" + boundary
    else:
        body = None
    try:
        with urllib.request.urlopen(
            urllib.request.Request(base + route, data=body, headers=headers), timeout=60
        ) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def verify(image, checkpoint, release, fixture, out):
    out.mkdir(parents=True, exist_ok=False)
    image_id = json.loads(docker("image", "inspect", image))[0]["Id"]
    content = fixture.read_bytes()
    suffix = uuid.uuid4().hex[:12]
    owned, active = [], []
    resources = [
        "--memory",
        "1g",
        "--cpus",
        "2",
        "--pids-limit",
        "128",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=268435456",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
    ]
    mounts = [
        "--mount",
        f"type=bind,src={checkpoint.resolve()},dst=/run/model.pt,readonly",
        "--mount",
        f"type=bind,src={release.resolve()},dst=/run/model.release.json,readonly",
        "--mount",
        f"type=bind,src={fixture.resolve()},dst=/run/fixture.png,readonly",
    ]
    record = {
        "schema_version": 1,
        "scope": "Generated single-image container integration; no quality/browser/deployment claim",
        "image_id": image_id,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "release_sha256": hashlib.sha256(release.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(content).hexdigest(),
        "containers": [],
    }

    def save(name, payload):
        (out / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def capture(container):
        data = inspect(container)
        config = data["HostConfig"]
        assert config["Memory"] == 1073741824 and config["NanoCpus"] == 2000000000
        assert config["ReadonlyRootfs"]
        assert all(
            not mount["RW"] for mount in data["Mounts"] if mount["Type"] == "bind"
        )
        assert not any(
            mount["Destination"].endswith("docker.sock") for mount in data["Mounts"]
        )
        return {
            "id": data["Id"],
            "name": data["Name"].lstrip("/"),
            "image_id": data["Image"],
            "memory_limit_bytes": config["Memory"],
            "cpu_limit": config["NanoCpus"] / 1e9,
            "readonly_rootfs": config["ReadonlyRootfs"],
            "oom_killed": data["State"]["OOMKilled"],
            "exit_code": data["State"]["ExitCode"],
            "running": data["State"]["Running"],
            "mounts": [
                {
                    "destination": m["Destination"],
                    "type": m["Type"],
                    "read_only": not m["RW"],
                }
                for m in data["Mounts"]
            ],
        }

    def start_http(name, available=True):
        args = [
            "run",
            "--detach",
            "--name",
            name,
            *resources,
            "--publish",
            "127.0.0.1::8000",
            "--env",
            "ROOF_MAX_UPLOAD_BYTES=2048",
            "--env",
            "ROOF_MAX_BATCH_IMAGES=1",
        ]
        if available:
            args += mounts
        else:
            args += [
                "--env",
                "MODEL_PATH=/run/missing.pt",
                "--env",
                "ROOF_MODEL_RELEASE=/run/missing.json",
            ]
        container = docker(*args, image_id)
        assert re.fullmatch(r"[a-f0-9]{64}", container)
        owned.append(container)
        active.append(container)
        bindings = inspect(container)["NetworkSettings"]["Ports"]["8000/tcp"]
        assert len(bindings) == 1 and bindings[0]["HostIp"] == "127.0.0.1"
        base = "http://127.0.0.1:" + bindings[0]["HostPort"]
        for _ in range(180):
            try:
                if request(base, "/api/health/live")[0] == 200:
                    return container, base
            except (OSError, ValueError):
                pass
            if not inspect(container)["State"]["Running"]:
                raise RuntimeError("Container exited before liveness")
            time.sleep(0.25)
        raise RuntimeError("Container liveness deadline exceeded")

    try:
        cli_name = "roof-wave2-cli-" + suffix
        cli_code = (
            "import base64,json;from pathlib import Path;"
            "from roof_pipeline.inference_service import predict_file;"
            "r=predict_file('/run/model.pt','/run/model.release.json','/run/fixture.png','/tmp/proof');"
            "r['mask_base64']=base64.b64encode(Path('/tmp/proof/mask.png').read_bytes()).decode();"
            "r['probability_base64']=base64.b64encode(Path('/tmp/proof/probability.npy').read_bytes()).decode();"
            "print(json.dumps(r))"
        )
        cli = json.loads(
            docker(
                "run",
                "--name",
                cli_name,
                *resources,
                "--network",
                "none",
                *mounts,
                image_id,
                "python",
                "-c",
                cli_code,
            )
        )
        owned.append(cli_name)
        cli_mask = base64.b64decode(cli.pop("mask_base64"), validate=True)
        cli_probability = base64.b64decode(cli.pop("probability_base64"), validate=True)
        (out / "cli-mask.png").write_bytes(cli_mask)
        (out / "cli-probability.npy").write_bytes(cli_probability)
        save("cli-manifest.json", cli)
        assert cli["model"]["runtime"]["profile_matches"]

        container, base = start_http("roof-wave2-http-" + suffix)
        status, ready = request(base, "/api/health/ready")
        assert status == 200 and ready["supported_profile_ready"]
        save("readiness.json", ready)
        status, result = request(base, "/api/inference", content)
        assert status == 200
        mask = base64.b64decode(result.pop("mask_base64"), validate=True)
        probability = base64.b64decode(result.pop("probability_base64"), validate=True)
        assert probability == cli_probability and mask == cli_mask
        assert result["input_sha256"] == record["input_sha256"]
        assert (
            hashlib.sha256(probability).hexdigest()
            == result["artifacts"]["probability"]["sha256"]
        )
        assert result["model"]["checkpoint_sha256"] == record["checkpoint_sha256"]
        assert result["model"]["model_release_sha256"] == record["release_sha256"]
        (out / "http-mask.png").write_bytes(mask)
        (out / "http-probability.npy").write_bytes(probability)
        save("response.json", result)
        assert request(base, "/api/inference", b"invalid image")[0] == 400
        assert request(base, "/api/inference", b"x" * 4096)[0] == 413
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(
                pool.map(lambda _: request(base, "/api/inference", content), range(2))
            )
        statuses = sorted(status for status, _ in responses)
        assert statuses == [200, 503], statuses
        record["checks"] = {
            "cli_http_artifacts_identical": True,
            "supported_linux_profile_ready": True,
            "corrupt_upload_status": 400,
            "over_byte_limit_status": 413,
            "concurrent_request_statuses": statuses,
        }
        record["runtime"] = ready["model"]["runtime"]
        docker("stop", "--time", "10", container)
        active.remove(container)
        container, base = start_http("roof-wave2-missing-" + suffix, available=False)
        assert request(base, "/api/health/ready")[0] == 503
        assert request(base, "/api/inference", content)[0] == 503
        record["checks"]["missing_weights_readiness_and_inference_status"] = 503
        record["status"] = "passed"
    finally:
        for container in active:
            if re.fullmatch(r"[a-f0-9]{64}", container):
                docker("stop", "--time", "10", container)
        for container in owned:
            record["containers"].append(capture(container))
        save("proof.json", record)
    assert all(
        not row["running"] and not row["oom_killed"] for row in record["containers"]
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "model-release", "fixture", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    result = verify(
        args.image, args.checkpoint, args.model_release, args.fixture, args.out
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "image_id": result["image_id"],
                "checks": result["checks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
