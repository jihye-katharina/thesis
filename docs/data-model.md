# Transcript → Segment → Unit

The hierarchy stores each fact at its source:

| Object | Fields | Scope of its ID |
| --- | --- | --- |
| `Transcript` | `id`, `segments`, full text and optional metadata | One lecture; input `video_id` is preferred |
| `Segment` | `id`, `start`, `end`, original `text`, `units` | Unique within its transcript |
| `Unit` | `id`, `char_start`, `char_end`, `text` | Unique within its parent segment |

Default segment IDs remain `s00000`, `s00001`, etc., unless supplied by the input.
Unit IDs are `u0`, `u1`, etc., restarting at `u0` for each segment. Splitting an
oversized sentence also continues this numbering inside the same parent.

## Python usage

After parsing the input JSON as a `Transcript`, define the model input limits:

```python
constraints = [
    (embedder.count_tokens, embedder.input_limit),
    (summarizer.count_tokens, 512),
]

# Split one segment into units.
units = make_units(transcript.segments[0], constraints)

# Or build the complete hierarchy without mutating the original transcript.
prepared = transcript.with_units(constraints)
segment = prepared.segments[0]
unit = segment.units[0]

print(prepared.id, segment.id, unit.id)
print(segment.start, segment.end)  # Original segment timing.
assert segment.text[unit.char_start:unit.char_end] == unit.text
```

`Unit` has no `segment_id`, `segment_index`, `start`, or `end` fields. Access
those facts through its parent segment. Character offsets use Python string
indices, with an inclusive start and exclusive end. Sentence timing cannot be
inferred from these offsets: only the original segment interval is known.

The pipeline temporarily flattens ordered `(segment, unit)` pairs for embedding
and chunking. Embedding rows are indexed by their list position, so two units
called `u0` cannot overwrite one another. This temporary processing order is
not stored as a field on `Unit`.

## JSON output and references

New outputs use SEDL schema `2.0`. The following is a shortened illustration:

```json
{
  "schema_version": "2.0",
  "transcript_id": "lecture-01",
  "source_segments": [
    {
      "id": "s00000", "start": 0.0, "end": 10.0,
      "text": "Hello. World.",
      "units": [
        {"id": "u0", "char_start": 0, "char_end": 6, "text": "Hello."},
        {"id": "u1", "char_start": 7, "char_end": 13, "text": "World."}
      ]
    },
    {
      "id": "s00001", "start": 10.0, "end": 20.0,
      "text": "Next.",
      "units": [
        {"id": "u0", "char_start": 0, "char_end": 5, "text": "Next."}
      ]
    }
  ]
}
```

Because a summary chunk may span several segments, its `source_unit_refs` is a
list of explicit pairs, for example:

```json
[
  {"segment_id": "s00000", "unit_id": "u1"},
  {"segment_id": "s00001", "unit_id": "u0"}
]
```

Evidence entries contain these same reference fields and their selection scores.
They do not repeat source text or timestamps; the audit and judge exporters
resolve the references. Boundary decisions use `before_unit_ref`; manual topic
annotations list `boundary_unit_refs`. The document supplies the transcript ID.
If combining references from several lectures, keep that transcript ID too:
`(transcript_id, segment_id, unit_id)` is the complete identity.

## Migration from earlier code

- Call `make_units(segment, constraints)` for one segment instead of passing the
  whole transcript. Use `transcript.with_units(constraints)` for all segments.
- Replace top-level `source_units` reads with traversal of
  `source_segments[].units`. The statistics field `source_units` remains a count.
- Replace `source_unit_ids` with `source_unit_refs`, and use both reference fields
  as a lookup key. The shared helpers are in `src/sedl/references.py`.
- For newly generated outputs, create a fresh boundary annotation template using
  `export-boundaries`. The unitization hash changes with the new representation;
  old global unit IDs cannot be silently applied to new local unit IDs.
- Readers still accept existing SEDL 1.x files and their corresponding original
  boundary annotations. Existing source JSON transcripts need no changes.

Neither reordering units nor replacing a reference with another segment's `u0`
is a valid migration. Preserve source order and verify exact parent text and
character offsets when importing any existing manual annotations.
