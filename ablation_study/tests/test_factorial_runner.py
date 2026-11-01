"""Exercise subprocess orchestration without spending real training compute."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location(
    "factorial_runner_tests", SCRIPTS / "run_factorial.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def setup(monkeypatch, tmp_path, extra=()):
    repo = tmp_path / "repo"
    (repo / "roof_pipeline").mkdir(parents=True)
    (repo / "roof_pipeline/train.py").write_text('"--uniform-replacement"')
    (repo / "ablation_study/scripts").mkdir(parents=True)
    (repo / "ablation_study/experiment_plan.json").write_text("{}")
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.csv").write_text("frozen manifest")
    checkpoint = tmp_path / "initial.pt"
    checkpoint.write_bytes(b"frozen checkpoint")
    out = tmp_path / "study"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_factorial",
            "--repo",
            str(repo),
            "--data",
            str(data),
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(out),
            *extra,
        ],
    )
    monkeypatch.setattr(
        runner,
        "preflight",
        lambda *args: {
            "dataset_content_sha256": "data",
            "split_counts": {"train": 4736, "val": 1036, "test": 2416},
        },
    )
    return repo, data, checkpoint, out


def test_all_twelve_commands_are_matched_and_only_validate(monkeypatch, tmp_path):
    repo, data, checkpoint, out = setup(monkeypatch, tmp_path)
    calls = []
    frozen_calls = []
    monkeypatch.setattr(
        runner, "assert_frozen_inputs", lambda *args: frozen_calls.append(args)
    )
    monkeypatch.setattr(runner, "freeze_mining", lambda *args: None)
    sys.path.insert(0, str(SCRIPTS))
    import analyze_factorial

    verified = []
    monkeypatch.setattr(
        analyze_factorial, "read_run", lambda *args: verified.append(args)
    )

    def execute(command, cwd, logfile):
        assert (out / "registration.json").is_file()
        assert (out / "source_snapshot.zip").is_file()
        assert cwd == repo
        calls.append(command)
        if "roof_pipeline.hard_examples" in command:
            (out / "hard_examples.json").write_text("{}")

    monkeypatch.setattr(runner, "execute", execute)
    assert runner.main() == 0
    train = [c for c in calls if "roof_pipeline.train" in c]
    evaluate = [c for c in calls if "roof_pipeline.evaluate" in c]
    assert len(train) == len(evaluate) == len(verified) == 12
    assert len(frozen_calls) == 24
    for command in train:
        assert command[command.index("--init-checkpoint") + 1] == str(checkpoint)
        assert command[command.index("--epochs") + 1] == "20"
        assert command[command.index("--patience") + 1] == "21"
        assert ("--uniform-replacement" in command) != ("--sampling-report" in command)
    assert sum("--uniform-replacement" in c for c in train) == 6
    assert sum(c[c.index("--photometric") + 1] == "robust" for c in train) == 6
    assert all(c[c.index("--split") + 1] == "val" for c in evaluate)
    assert all(c[c.index("--checkpoint") + 1].endswith("final.pt") for c in evaluate)
    assert "analyze_factorial.py" in calls[-2][1]
    assert "plot_factorial.py" in calls[-1][1]
    assert (
        json.loads((out / "workflow_status.json").read_text())["completed_runs"] == 12
    )


@pytest.mark.parametrize(
    "extra", [("--lr", "nan"), ("--weight-decay", "inf"), ("--epochs", "0")]
)
def test_invalid_budget_rejected_without_output(monkeypatch, tmp_path, extra):
    _, _, _, out = setup(monkeypatch, tmp_path, extra)
    with pytest.raises(SystemExit, match="Invalid"):
        runner.main()
    assert not out.exists()


def test_source_drift_rejected(monkeypatch, tmp_path):
    repo, data, checkpoint, _ = setup(monkeypatch, tmp_path)
    args = runner.get_args()
    reg = runner.registration(args, data, checkpoint, repo)
    runner.assert_frozen_inputs(reg, repo, data, checkpoint)
    (repo / "roof_pipeline/train.py").write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        runner.assert_frozen_inputs(reg, repo, data, checkpoint)


def test_interruption_stops_later_runs(monkeypatch, tmp_path):
    _, _, _, out = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "freeze_mining", lambda *args: None)
    monkeypatch.setattr(runner, "assert_frozen_inputs", lambda *args: None)
    calls = []

    def execute(command, *args):
        calls.append(command)
        if "roof_pipeline.train" in command:
            raise RuntimeError("simulated training failure")

    monkeypatch.setattr(runner, "execute", execute)
    with pytest.raises(RuntimeError, match="simulated"):
        runner.main()
    assert len(calls) == 2
    assert not (out / "analysis/report.md").exists()


def test_frozen_mining_cannot_be_silently_regenerated(monkeypatch, tmp_path):
    _, data, checkpoint, out = setup(monkeypatch, tmp_path, ("--resume",))
    out.mkdir()
    reg = runner.registration(
        runner.get_args(),
        data,
        checkpoint,
        Path(sys.argv[sys.argv.index("--repo") + 1]),
    )
    (out / "registration.json").write_text(json.dumps(reg))
    (out / "mining_fingerprint.json").write_text('{"sha256":"previous"}')
    with pytest.raises(ValueError, match="Frozen mining report is missing"):
        runner.main()
