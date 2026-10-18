"""The CLI's text output."""

import json

import pytest

from kiseki import actions, cli


def test_machines_says_none_for_a_machine_without_a_gpu(monkeypatch, capsys):
    record = {"name": "local", "ssh": None, "gpu_awake": None, "runner": False}
    monkeypatch.setattr(actions, "machine_records", lambda: [record])

    assert cli.main(["machines"]) == 0

    assert capsys.readouterr().out.splitlines()[1].split() == ["local", "-", "none"]


def test_machines_config_reports_what_setup_needs_without_probing(home, monkeypatch, capsys):
    monkeypatch.setattr(actions, "machine_records", lambda: pytest.fail("probed the machines"))
    config = home.parent / "config"
    config.mkdir()
    (config / "machines.toml").write_text(
        '[contact]\nname = "Ada"\nemail = "ada@example.org"\n\n'
        '[machines.lugia]\nssh = "lugia"\nnotice = "/etc/motd.tail"\n\n'
        '[machines.vps]\nssh = "vps"\n'
    )

    assert cli.main(["machines", "--config", "--json"]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "path": str(config / "machines.toml"),
        "exists": True,
        "machines": ["lugia", "vps"],
        "notice": ["lugia"],
        "contact": True,
    }
