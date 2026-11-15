# Run the prepared comparison on a Linux GPU Pod

These helpers transfer the exact registered source/data and derive a fresh Linux
plan. They do not change the Windows plan, scientific settings, encoder tensors,
labels, geographic splits, or checkpoint selection. No account key is bundled.

From the repository root, build two archives in a fresh directory:

```powershell
.\.venv-gpu\Scripts\python.exe scripts/runpod/package_comparison.py --plan runs/whu_architecture_comparison_2026-11-14/plan.json --out runs/runpod_transfer_2026-11-14
```

`code.tar.gz` is small. `data.tar` streams the manifest, metadata and exactly the
manifest-listed image/mask files. Previews and other data-directory files are
excluded. SHA256SUMS and transfer_manifest.json describe both archive bytes.

Transfer to the Pod's SSH/SFTP endpoint using its verified host key. `sftp`'s
`reput` resumes an interrupted upload; remote SHA256 verification is mandatory
before extraction. Example commands (replace host/port with the actual Pod):

```text
sftp -P PORT -i C:/Users/tanne/.ssh/id_ed25519 root@HOST
mkdir /workspace/roof-transfer
cd /workspace/roof-transfer
reput runs/runpod_transfer_2026-11-14/code.tar.gz
reput runs/runpod_transfer_2026-11-14/data.tar
put runs/runpod_transfer_2026-11-14/SHA256SUMS
put runs/runpod_transfer_2026-11-14/transfer_manifest.json
```

On the Pod, preserve any existing directory by choosing a fresh destination:

```bash
cd /workspace/roof-transfer
sha256sum -c SHA256SUMS
mkdir /workspace/roof-comparison
tar -xzf code.tar.gz -C /workspace/roof-comparison
tar -xf data.tar -C /workspace/roof-comparison
cd /workspace/roof-comparison
nohup bash scripts/runpod/bootstrap.sh --out /workspace/roof-comparison/runs/whu_architecture_runpod_2026-11-14 > /workspace/roof-bootstrap.log 2>&1 &
```

This first form prepares and profiles only. The isolated environment pins
PyTorch2.5.1/cu121, torchvision0.20.1/cu121, SMP0.5.0 and the same core numerical
packages as the local GPU environment. An Ampere A5000 supports that CUDA build.
The derivative plan must reproduce all encoder/data/source fingerprints and
scientific controls; its actual Linux/Python/runtime environment is recorded.

For a single unattended prepare/profile/train flow, use a **fresh** output path
and append `--execute --hourly-rate QUOTED_TOTAL_RATE --max-estimated-cost APPROVED_CEILING
--setup-cost-allowance ALREADY_SPENT_ALLOWANCE --wall-timeout-seconds SECONDS`.
The cost gate uses the remote upper estimate before training. It is not a hard
provider billing cap: configure an external Runpod stop/termination control.
Process timeout kills the job group but cannot stop Pod billing by itself.

`remote_status.json`, remote_profile.log, remote_training.log and per-model logs
show progress. On completion or failure, a result tar and SHA256 sidecar preserve
available plans, code identities, checkpoints, histories, validation evidence and
logs. Download and verify those files before terminating storage. The tar itself
is outside the run directory and does not recursively include itself.

## Verified result retrieval

The local `watch_remote.py` checks status over pinned SSH, downloads the archive,
verifies its outer SHA256 and every registered member, safely extracts it, then
writes the matching retrieval ACK. The remote supervisor may remove the Pod only
after that ACK. A process timeout alone does not stop provider billing.

If large SFTP reads stall, the watcher can use the Pod's existing authenticated
Jupyter HTTPS file route for the archive while retaining SSH for status and ACK:

```text
--jupyter-connection <private-connection.json>
```

The private file contains the existing `base_url` and `token`; keep it outside
Git. Tokens are sent only in an Authorization header. TLS is verified, redirects
are rejected, resumed byte ranges and lengths are checked against SSH metadata,
and incomplete downloads never receive an ACK. The client environment needs
`paramiko` and, for this optional route, `requests`.

For a GPU study, keep the Pod in GPU mode. Do not substitute CPU mode to work
around transfer problems. When retiring or replacing a job, identify its entire
process group and verify that no fitting child survived before starting another
job or measuring throughput. Disable the old cleanup supervisor before allowing
an ACK for a superseded output to affect a Pod still in use.
