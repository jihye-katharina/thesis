from dataclasses import dataclass
import math
import re

@dataclass(frozen=True)
class Segment:
    id: str # s00000, s00001, ...
    start: float
    end: float
    text: str

@dataclass(frozen=True)
class Transcript:
    id: str # video_id
    segments: tuple[Segment, ...]
    text: str | None = None
    source: str | None = None
    language: str | None = None

    @classmethod
    def from_dict(cls, value: dict) -> "Transcript":
        if not isinstance(value, dict) or not isinstance(value.get("segments"), list):
            raise TypeError("Transcript must contain a segments array.")
        if not value["segments"]:
            raise ValueError("Transcript must have at least one segment.")

        full_text = value.get("text")
        if full_text is not None and not isinstance(full_text, str):
            raise TypeError("Optional full lecture text must be a string.")

        segments = []
        seen = set()
        for i, sgm in enumerate(value["segments"]):
            if not isinstance(sgm, dict):
                raise TypeError(f"Segment {i} must be an object.")

            times = [sgm.get(k) for k in ("start", "end")]
            if any(isinstance(t, bool) or not isinstance(t, (int, float)) or
                   not math.isfinite(t) for t in times):
                raise ValueError(f"Segment {i}: start/end must be finite seconds.")
            start, end = times
            if start < 0 or end <= start:
                raise ValueError(f"Segment {i}: require 0 <= start < end.")
            if segments and start < segments[-1].start:
                raise ValueError("Segments must be ordered by start time.")
            
            text = sgm.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Segment {i}: text must be nonempty.")
            
            identifier = str(sgm.get("id", f"s{i:05d}"))
            if identifier in seen:
                raise ValueError(f"Duplicate segment id: {identifier}")
            
            seen.add(identifier)
            segments.append(Segment(identifier, float(start), float(end), text))

        for key in ("source", "language"):
            metadata = value.get(key)
            if metadata is not None and (not isinstance(metadata, str) or not metadata.strip()):
                raise ValueError(f"Optional {key} must be a nonempty string.")
        
        return cls(
            id=str(value.get("video_id")),
            segments=tuple(segments),
            text=full_text,
            source=value.get("source"),
            language=value.get("language")
        )

@dataclass(frozen=True)
class Unit:
  id: str # unit_id; u0, u1, ...
  segment_id: str
  segment_start: float
  segment_end: float
  text: str

def make_units(transcript: Transcript) -> list[Unit]:
    """Split segment text into whole sentence-like units, without token splitting.
       IDs count u0, u1, ... across the whole transcript.
       Units stay in original order across time segments.
    """
    if not isinstance(transcript, Transcript):
        raise TypeError("Transcript is expected containing the lecture's segments")
    units = []
  
    # A deterministic English-oriented splitter
    # Use regular expressions to split the text into sentences based on punctuation followed by whitespace.
    for segment in transcript.segments:
      sentences = re.split(r'(?<=[.!?])\s+|\n+', segment.text)

      for s in sentences:
        s = s.strip()
        if s:
          units.append(
              Unit(id=f"u{len(units)}",
                   segment_id=segment.id,
                   segment_start=segment.start,
                   segment_end=segment.end,
                   text=s)
          )

    return units
    
@dataclass(frozen=True)
class Chunk:
  id: str
  units: tuple[Unit, ...]

  @property
  def text(self) -> str:
    return "\n".join(u.text for u in self.units)
