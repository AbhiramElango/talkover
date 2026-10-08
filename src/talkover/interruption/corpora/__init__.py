"""Corpus readers that produce timed, dialogue-act-labelled meetings."""

from typing import Iterator, Protocol, runtime_checkable

from talkover.interruption.corpora.ami import AmiCorpusReader
from talkover.interruption.events import Meeting


@runtime_checkable
class CorpusReader(Protocol):
    def meetings(self) -> Iterator[Meeting]: ...


__all__ = ["AmiCorpusReader", "CorpusReader"]
