"""Local boundary evaluation, and data export for external judge frameworks."""

import math

from .references import (entry_unit_refs, evidence_unit_ref, index_source_units,
                         iter_source_units, reference_key)


def boundary_f1(predicted, reference, tolerance_seconds=5.0):
    """Maximum one-to-one matching of sorted temporal boundaries within tolerance.

    Endpoints are excluded by the caller. Empty/empty is perfect; one empty side
    yields zero F1. Never count one annotation more than once.
    """
    if not math.isfinite(tolerance_seconds) or tolerance_seconds < 0:
        raise ValueError("Boundary tolerance must be finite and nonnegative")
    for values in (predicted, reference):
        if any(isinstance(t, bool) or not isinstance(t, (int, float)) or
               not math.isfinite(t) or t < 0 for t in values):
            raise ValueError("Boundaries must be finite nonnegative seconds")
        if len(set(values)) != len(values):
            raise ValueError("Boundary times must be unique")
    predicted, reference = sorted(predicted), sorted(reference)
    i = j = matches = 0
    while i < len(predicted) and j < len(reference):
        if abs(predicted[i] - reference[j]) <= tolerance_seconds:
            matches += 1
            i += 1
            j += 1
        elif predicted[i] < reference[j]:
            i += 1
        else:
            j += 1
    precision = matches / len(predicted) if predicted else float(not reference)
    recall = matches / len(reference) if reference else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "matched": matches,
            "predicted": len(predicted), "reference": len(reference),
            "tolerance_seconds": tolerance_seconds}


def boundary_annotation_template(sedl):
    # Parent context is retained without exposing model-predicted boundaries.
    segments = [{key: segment[key] for key in ("id", "start", "end", "text")}
                for segment in sedl["source_segments"]]
    units = [{"id": unit["id"], "segment_id": segment["id"], "char_start": unit["char_start"],
              "char_end": unit["char_end"], "text": unit["text"]}
             for _, segment, unit in iter_source_units(sedl)]
    return {"transcript_id": sedl["transcript_id"],
            "unitization_sha256": sedl["unitization_sha256"],
            "annotation_status": "draft", "boundary_unit_refs": [],
            "instructions": "Read the ordered units without model boundary predictions. "
                            "List {segment_id, unit_id} for the first unit after each topic change, "
                            "excluding lecture start. Mark annotation_status complete after review.",
            "source_segments": segments, "source_units": units}


def segmentation_errors(predicted, reference, unit_count):
    """Pk and WindowDiff on common units; k uses the reference mean segment size.

    An index b denotes a boundary immediately BEFORE unit b. Windows compare
    endpoints i and i+k, counting boundaries i < b <= i+k. Lower is better.
    """
    if unit_count < 2:
        return {"pk": None, "window_diff": None, "window_units": None,
                "note": "At least two units are required"}
    k = min(unit_count - 1, max(1, round(unit_count / (2 * (len(reference) + 1)))))
    pk_errors = wd_errors = 0
    for i in range(unit_count - k):
        p = sum(i < b <= i + k for b in predicted)
        r = sum(i < b <= i + k for b in reference)
        pk_errors += (p == 0) != (r == 0)
        wd_errors += p != r
    return {"pk": pk_errors / (unit_count - k),
            "window_diff": wd_errors / (unit_count - k), "window_units": k}


def evaluate_boundaries(sedl, annotation, tolerance_seconds=5.0, tolerance_units=1):
    if annotation.get("transcript_id") != sedl["transcript_id"]:
        raise ValueError("Annotation transcript_id does not match SEDL")
    lookup = index_source_units(sedl)
    if "boundary_unit_refs" in annotation or "boundary_unit_ids" in annotation:
        if annotation.get("annotation_status") != "complete":
            raise ValueError("Unit boundary annotations must be marked complete after human review")
        if annotation.get("unitization_sha256") != sedl["unitization_sha256"]:
            raise ValueError("Annotation unitization_sha256 does not match SEDL")
        if isinstance(tolerance_units, bool) or not isinstance(tolerance_units, int) or tolerance_units < 0:
            raise ValueError("tolerance_units must be a nonnegative integer")
        indices = {key: i for i, key in enumerate(lookup)}
        if "boundary_unit_refs" in annotation:
            reference = annotation["boundary_unit_refs"]
            if not isinstance(reference, list):
                raise ValueError("boundary_unit_refs must be a list of {segment_id, unit_id} references")
            keys = [reference_key(ref) for ref in reference]
        else:
            if "source_units" not in sedl:
                raise ValueError("Use boundary_unit_refs for segment-local IDs; regenerate the annotation template")
            reference = annotation["boundary_unit_ids"]
            if not isinstance(reference, list) or not all(isinstance(i, str) for i in reference):
                raise ValueError("Legacy boundary_unit_ids must be a list of global IDs")
            old_ids = {unit["id"]: key for key, (_, unit) in lookup.items()}
            if any(i not in old_ids for i in reference):
                raise ValueError("Unknown legacy boundary unit ID")
            keys = [old_ids[i] for i in reference]
        if any(key not in indices or indices[key] == 0 for key in keys):
            raise ValueError("Boundary references must identify known units excluding lecture start")
        predicted = [indices[reference_key(entry_unit_refs(sedl, e)[0])] for e in sedl["entries"][1:]]
        score = boundary_f1(predicted, [indices[key] for key in keys], tolerance_units)
        score["tolerance_units"] = score.pop("tolerance_seconds")
        return {"transcript_id": sedl["transcript_id"], "coordinate_system": "unit_index",
                "boundary_f1": score,
                "segmentation_errors": segmentation_errors(
                    predicted, [indices[key] for key in keys], len(indices)),
                "note": "Common unit indices preserve distinct cuts inside the same source segment."}
    if "boundaries" not in annotation or not isinstance(annotation["boundaries"], list):
        raise ValueError("Annotation must contain a boundaries array in seconds")
    # Use the first unit's PARENT segment start, not chunk interval minima (which can
    # overlap for split/overlapping source segments). Repeated times collapse.
    predicted = sorted(set(lookup[reference_key(entry_unit_refs(sedl, entry)[0])][0]["start"]
                           for entry in sedl["entries"][1:]))
    split_inside_segment = any(entry["boundary"]["start"].get("at_char", 0) > 0
                               for entry in sedl["entries"][1:])
    return {"transcript_id": sedl["transcript_id"],
            "boundary_f1": boundary_f1(predicted, annotation["boundaries"], tolerance_seconds),
            "has_within_segment_cuts": split_inside_segment,
            "timing_note": "Within-segment cuts inherit segment start times; "
                           "review these before interpreting timestamp boundary F1."}


def judge_records(sedl):
    """Export inputs only. These records are NOT G-Eval or RAGAs scores."""
    lookup = index_source_units(sedl)
    for entry in sedl["entries"]:
        yield {
            "transcript_id": sedl["transcript_id"], "chunk_id": entry["id"],
            "response": entry["summary"],
            "retrieved_contexts": [lookup[reference_key(evidence_unit_ref(item))][1]["text"]
                                   for item in entry["evidence"]],
            "source_chunk": "\n".join(lookup[reference_key(ref)][1]["text"]
                                       for ref in entry_unit_refs(sedl, entry)),
            "note": "Unscored export. Choose and document a G-Eval/RAGAs judge and rubric.",
        }
