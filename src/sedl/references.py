"""Resolve source unit references within one SEDL transcript.

The containing SEDL supplies transcript_id. A reference is the pair
{segment_id, unit_id}. Current IDs are unique within the lecture; segment_id
retains the original time lookup. Readers also support flat 1.x and nested 2.0
exports, whose local unit IDs require both reference fields.
"""


def unit_ref(segment_id, unit_id):
    return {"segment_id": segment_id, "unit_id": unit_id}


def reference_key(reference):
    if (not isinstance(reference, dict) or
            not isinstance(reference.get("segment_id"), str) or
            not isinstance(reference.get("unit_id"), str)):
        raise ValueError("A unit reference must contain string segment_id and unit_id")
    return reference["segment_id"], reference["unit_id"]


def iter_source_units(sedl):
    """Yield (reference, parent segment, unit) in original text order."""
    if "source_units" in sedl:  # Current 3.0 and legacy 1.x flat exports.
        segments = {s["id"]: s for s in sedl["source_segments"]}
        for unit in sedl["source_units"]:
            segment = segments[unit["segment_id"]]
            yield unit_ref(segment["id"], unit["id"]), segment, unit
    else:
        for segment in sedl["source_segments"]:
            for unit in segment["units"]:
                yield unit_ref(segment["id"], unit["id"]), segment, unit


def index_source_units(sedl):
    result = {}
    seen_ids = set()
    for reference, segment, unit in iter_source_units(sedl):
        key = reference_key(reference)
        if key in result:
            raise ValueError(f"Duplicate unit reference: {reference}")
        if "source_units" in sedl and unit["id"] in seen_ids:
            raise ValueError(f"Duplicate lecture-wide unit ID: {unit['id']}")
        seen_ids.add(unit["id"])
        result[key] = (segment, unit)
    return result


def entry_unit_refs(sedl, entry):
    if "source_unit_refs" in entry:
        return entry["source_unit_refs"]
    old_units = {u["id"]: u for u in sedl["source_units"]}
    return [unit_ref(old_units[i]["segment_id"], i) for i in entry["source_unit_ids"]]


def evidence_unit_ref(evidence):
    # 1.x evidence records used id; later records contain references only.
    identifier = evidence["unit_id"] if "unit_id" in evidence else evidence["id"]
    return unit_ref(evidence["segment_id"], identifier)
