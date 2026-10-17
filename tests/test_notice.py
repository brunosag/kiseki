from datetime import UTC, datetime

from kiseki import notice
from kiseki.machines import Contact

CONTACT = Contact("Ada Lovelace", "ada@example.org")


def test_forecast_pads_and_rounds_up_to_half_hours():
    now = datetime(2026, 10, 17, 15, 0, tzinfo=UTC)
    # 2 h remaining: 2 h · 1.1 + 30 min = 2 h 42 min -> 17:42, rounded up to 18:00.
    assert notice.forecast_end(now, 7200) == datetime(2026, 10, 17, 18, 0, tzinfo=UTC)
    # nothing remaining: 30 min margin -> exactly 15:30.
    assert notice.forecast_end(now, 0) == datetime(2026, 10, 17, 15, 30, tzinfo=UTC)


def test_block_names_person_email_and_interval_in_brasilia_time():
    start = datetime(2026, 10, 17, 15, 0, tzinfo=UTC)
    end = datetime(2026, 10, 17, 18, 0, tzinfo=UTC)
    block = notice.render(CONTACT, start, end, running=1, queued=2, now=start, user="ada")
    assert "Ada Lovelace <ada@example.org>" in block
    assert "de 17/10/2026 12:00 até a previsão de 17/10/2026 15:00" in block  # UTC-3
    assert "1 rodando, 2 na fila" in block
    assert block.splitlines()[0] == "# >>> kiseki ada"


def test_write_replaces_then_removes_only_our_block(tmp_path):
    path = tmp_path / "motd.tail"
    path.write_text("Regras do servidor.\n\n# >>> kiseki bob\nBob's runs\n# <<< kiseki bob\n")
    original = path.read_text()
    first = "# >>> kiseki ada\nfirst\n# <<< kiseki ada"
    second = "# >>> kiseki ada\nsecond\n# <<< kiseki ada"
    assert notice.write(path, first, user="ada")
    assert notice.write(path, second, user="ada")
    text = path.read_text()
    assert "second" in text and "first" not in text and "Bob's runs" in text
    assert not notice.write(path, second, user="ada")  # unchanged
    assert notice.write(path, None, user="ada")
    assert path.read_text() == original
