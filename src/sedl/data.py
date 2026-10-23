"""Validated transcript input and conservative sentence-level provenance."""

from copy import deepcopy
from dataclasses import dataclass, replace
import math
import re
from typing import Callable


@dataclass(frozen=True)
class Unit:
    """A piece of one segment; its ID is local to that parent segment."""

    id: str
    char_start: int
    char_end: int
    text: str


@dataclass(frozen=True)
class Segment:
    id: str
    start: float
    end: float
    text: str
    units: tuple[Unit, ...] = ()


@dataclass(frozen=True)
class Transcript:
    id: str
    segments: tuple[Segment, ...]
    text: str | None = None
    source: str | None = None
    language: str | None = None
    provenance: dict | None = None

    def with_units(self, constraints: list[tuple[Callable[[str], int], int]]) -> "Transcript":
        """Return a copy with units nested under each segment; preserve the input."""
        return replace(self, segments=tuple(
            replace(segment, units=tuple(make_units(segment, constraints)))
            for segment in self.segments
        ))

    @classmethod
    def from_dict(cls, value: dict) -> "Transcript":
        if not isinstance(value, dict) or not isinstance(value.get("segments"), list):
            raise ValueError("Transcript must contain a segments array.")
        if not value["segments"]:
            raise ValueError("Transcript must have at least one segment.")
        segments = []
        seen = set()
        for i, row in enumerate(value["segments"]):
            if not isinstance(row, dict):
                raise ValueError(f"Segment {i} must be an object.")
            times = [row.get(k) for k in ("start", "end")]
            if any(isinstance(t, bool) or not isinstance(t, (int, float)) or
                   not math.isfinite(t) for t in times):
                raise ValueError(f"Segment {i}: start/end must be finite seconds.")
            start, end = times
            if start < 0 or end <= start:
                raise ValueError(f"Segment {i}: require 0 <= start < end.")
            if segments and start < segments[-1].start:
                raise ValueError("Segments must be ordered by start time; input is not sorted automatically.")
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Segment {i}: text must be nonempty.")
            identifier = str(row.get("id", f"s{i:05d}"))
            if identifier in seen:
                raise ValueError(f"Duplicate segment id: {identifier}")
            seen.add(identifier)
            segments.append(Segment(identifier, float(start), float(end), text))
        full_text = value.get("text")
        if full_text is not None and not isinstance(full_text, str):
            raise ValueError("Optional full lecture text must be a string.")
        for key in ("source", "language"):
            metadata = value.get(key)
            if metadata is not None and (not isinstance(metadata, str) or not metadata.strip()):
                raise ValueError(f"Optional {key} must be a nonempty string.")
        provenance = value.get("provenance")
        if provenance is not None and not isinstance(provenance, dict):
            raise ValueError("Optional provenance must be an object.")
        return cls(
            id=str(value.get("video_id", value.get("id", "lecture"))),
            segments=tuple(segments),
            text=full_text,
            source=value.get("source"),
            language=value.get("language"),
            provenance=deepcopy(provenance),
        )


def make_units(segment: Segment,
               constraints: list[tuple[Callable[[str], int], int]]) -> list[Unit]:
    """Split ONE segment into units numbered u0, u1, ... within that segment.

    Offsets are Python character offsets in the ORIGINAL segment text. Parent
    identity and timestamps remain on Segment rather than being copied into Unit.
    Each constraint counts tokens without special tokens and supplies a capacity.
    """
    if not isinstance(segment, Segment):
        raise TypeError("make_units expects one Segment; use transcript.with_units for the whole lecture")
    units = []
    offset = 0
    # A deterministic English-oriented splitter; abbreviations may over-split.
    ends = [m.end() for m in re.finditer(r'(?<=[.!?])\s+|\n+', segment.text)]
    ends.append(len(segment.text))
    for end in ends:
        left, right = offset, end
        offset = end
        while left < right and segment.text[left].isspace():
            left += 1
        while right > left and segment.text[right - 1].isspace():
            right -= 1
        while left < right:
            remaining = segment.text[left:right]
            if all(count(remaining) <= limit for count, limit in constraints):
                cut = right
            else:
                # Find a fitting prefix, then prefer a nearby word boundary.
                lo, hi = left, right
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if all(count(segment.text[left:mid]) <= limit
                           for count, limit in constraints):
                        lo = mid
                    else:
                        hi = mid - 1
                cut = lo
                word_break = segment.text.rfind(" ", left + 1, cut)
                if word_break > left:
                    cut = word_break
                if cut <= left:
                    raise ValueError("Token budget cannot fit a source character.")
                # Token counts can be non-monotone at subword boundaries.
                while cut > left and any(count(segment.text[left:cut]) > limit
                                         for count, limit in constraints):
                    cut -= 1
                if cut <= left:
                    raise ValueError("Token budget cannot fit a source character.")
            piece_end = cut
            while piece_end > left and segment.text[piece_end - 1].isspace():
                piece_end -= 1
            if piece_end > left:
                units.append(Unit(f"u{len(units)}", left, piece_end, segment.text[left:piece_end]))
            left = cut
            while left < right and segment.text[left].isspace():
                left += 1
    return units
