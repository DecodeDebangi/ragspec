"""Self-contained HTML dashboard for one or more runs."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

from .models import Run


def render_report(runs: list[Run], title: str = "ragspec report") -> str:
    template = files("ragspec").joinpath("templates/report.html").read_text()
    data = json.dumps([r.model_dump(mode="json") for r in runs], separators=(",", ":"))
    # Keep the JSON from closing the <script> tag it is embedded in.
    data = data.replace("</", "<\\/")
    return template.replace("__TITLE__", title).replace("__DATA__", data)


def write_report(runs: list[Run], path: Path, title: str = "ragspec report") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(runs, title))
    return path
