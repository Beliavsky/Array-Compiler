"""External corpus discovery for Array Compiler.

The compiler repo keeps only a small set of active corpus selections locally.
Large example sets live in a separate repository, currently
`Pure-Fortran-Examples`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "corpus.toml"


@dataclass(frozen=True)
class CorpusRoot:
    name: str
    path: Path


@dataclass(frozen=True)
class CorpusCase:
    corpus: str
    language: str
    path: Path
    tags: tuple[str, ...] = ()


def load_corpus_config(config_path: Path | None = None) -> dict:
    path = config_path or DEFAULT_CONFIG_PATH
    with path.open("rb") as handle:
        return tomllib.load(handle)


def configured_roots(config_path: Path | None = None) -> dict[str, CorpusRoot]:
    config = load_corpus_config(config_path)
    roots: dict[str, CorpusRoot] = {}
    for entry in config.get("corpus", []):
        root_path = Path(entry["path"]).expanduser()
        roots[entry["name"]] = CorpusRoot(name=entry["name"], path=root_path)
    return roots


def configured_cases(config_path: Path | None = None) -> list[CorpusCase]:
    config = load_corpus_config(config_path)
    roots = configured_roots(config_path)
    cases: list[CorpusCase] = []
    for entry in config.get("case", []):
        corpus_name = entry["corpus"]
        root = roots[corpus_name]
        rel_path = Path(entry["path"])
        cases.append(
            CorpusCase(
                corpus=corpus_name,
                language=entry["language"],
                path=root.path / rel_path,
                tags=tuple(entry.get("tags", [])),
            )
        )
    return cases


def discover_language_files(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for suffix in suffixes:
        out.extend(sorted(root.rglob(f"*{suffix}")))
    return out
