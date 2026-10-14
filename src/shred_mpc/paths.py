"""Resolve checkout assets without depending on the original workspace path."""
from pathlib import Path


def repository_root():
    for start in (Path.cwd(), Path(__file__).resolve().parents[2]):
        for path in (start, *start.parents):
            marker=path/'pyproject.toml'
            if marker.is_file() and 'shred-mpc-research' in marker.read_text():
                return path
    return Path.cwd()
