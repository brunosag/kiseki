"""The shared-server notice: a block in a world-writable file (lugia's /etc/motd.tail) that
says who is running experiments and for how long. That server's rule: anyone who starts an
experiment must write "nome, e-mail, e previsão do intervalo de uso do servidor (data de
início e previsão de final)". The queue runner keeps the block current while runs are
active (`kiseki.runner`) and removes it when the queue empties.

The block sits between marker lines naming the Unix user; the rest of the file is kept.
"""

import fcntl
import getpass
import math
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from kiseki.machines import Contact

ZONE = ZoneInfo("America/Sao_Paulo")  # the servers' users read Brasília time (UTC-3)
MARGIN = 0.1  # forecasts are padded by 10% of the remaining time, at least 30 minutes


def _markers(user: str) -> tuple[str, str]:
    return f"# >>> kiseki {user}", f"# <<< kiseki {user}"


def forecast_end(now: datetime, remaining_seconds: float) -> datetime:
    padded = remaining_seconds * (1 + MARGIN) + 1800
    end = now + timedelta(seconds=padded)
    minutes = math.ceil((end.minute + end.second / 60) / 30) * 30  # round up to :00 or :30
    return end.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minutes)


def render(
    contact: Contact,
    start: datetime,
    end: datetime,
    running: int,
    queued: int,
    now: datetime,
    user: str | None = None,
) -> str:
    def when(moment: datetime) -> str:
        return moment.astimezone(ZONE).strftime("%d/%m/%Y %H:%M")

    begin, finish = _markers(user or getpass.getuser())
    runs = f"{running} rodando" + (f", {queued} na fila" if queued else "")
    return "\n".join(
        [
            begin,
            f"Experimentos em execução na GPU (kiseki): {contact.name} <{contact.email}>",
            f"Uso do servidor: de {when(start)} até a previsão de {when(end)} "
            "(horário de Brasília)",
            f"Execuções: {runs}. Atualizado em {when(now)}.",
            finish,
        ]
    )


def replace_block(text: str, block: str | None, user: str | None = None) -> str:
    """`text` with our block replaced by `block` (or removed when None)."""
    begin, finish = _markers(user or getpass.getuser())
    lines = text.splitlines()
    if begin in lines and finish in lines[lines.index(begin) :]:
        start = lines.index(begin)
        end = lines.index(finish, start)
        lines = lines[:start] + lines[end + 1 :]
    while lines and not lines[-1].strip():
        lines.pop()
    if block:
        lines += ["", *block.splitlines()]
    return "\n".join(lines) + "\n"


def write(path: Path, block: str | None, user: str | None = None) -> bool:
    """Put our block in the file (or take it out). The file is rewritten in place: its
    directory (/etc) is not writable, and other users may hold it open. Returns whether
    the file changed."""
    with path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        text = handle.read()
        updated = replace_block(text, block, user)
        if updated == text:
            return False
        handle.seek(0)
        handle.write(updated)
        handle.truncate()
        return True
