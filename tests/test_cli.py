"""The CLI's text output."""

from kiseki import actions, cli


def test_machines_says_none_for_a_machine_without_a_gpu(monkeypatch, capsys):
    record = {"name": "local", "ssh": None, "gpu_awake": None, "runner": False}
    monkeypatch.setattr(actions, "machine_records", lambda: [record])

    assert cli.main(["machines"]) == 0

    assert capsys.readouterr().out.splitlines()[1].split() == ["local", "-", "none"]
