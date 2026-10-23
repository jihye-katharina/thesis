"""Human evidence-support review; export never supplies invented ratings."""

import csv
import json
import re

from .references import entry_unit_refs, evidence_unit_ref, index_source_units, reference_key


LABELS = {"supported", "unsupported", "unclear"}
FIELDS = ["transcript_id", "source_sha256", "chunk_id", "sentence_index", "summary_sentence",
          "source_interval_seconds", "source_chunk", "selected_evidence", "source_support",
          "evidence_support", "rater", "notes"]


def export_audit(sedl, path):
    lookup = index_source_units(sedl)
    with open(path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for entry in sedl["entries"]:
            # Sentence heuristic only: a reviewer may need to split compound claims.
            sentences = re.split(r"(?<=[.!?])\s+", entry["summary"].strip())
            for i, sentence in enumerate(sentences):
                writer.writerow({
                    "transcript_id": sedl["transcript_id"],
                    "source_sha256": sedl["source_sha256"],
                    "chunk_id": entry["id"], "sentence_index": i,
                    "summary_sentence": sentence,
                    "source_interval_seconds": f"{entry['start']}..{entry['end']}",
                    "source_chunk": "\n".join(lookup[reference_key(ref)][1]["text"]
                                               for ref in entry_unit_refs(sedl, entry)),
                    "selected_evidence": "\n".join(
                        f"[{json.dumps(evidence_unit_ref(item), ensure_ascii=False)}] "
                        + lookup[reference_key(evidence_unit_ref(item))][1]["text"]
                        for item in entry["evidence"]),
                })


def score_audit(path):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("Expected an evidence audit CSV exported by export-audit")
        rows = list(reader)
    results = {}
    for field in ("source_support", "evidence_support"):
        values = [row[field].strip().lower() for row in rows]
        if any(v and v not in LABELS for v in values):
            raise ValueError(f"{field} must be blank, supported, unsupported, or unclear")
        counts = {label: values.count(label) for label in sorted(LABELS)}
        determinate = counts["supported"] + counts["unsupported"]
        results[field] = {**counts, "unrated": values.count(""),
                          "supported_fraction_determinate": (counts["supported"] / determinate
                                                              if determinate else None),
                          "determinate_coverage": determinate / len(rows) if rows else None}
    return {"rows": len(rows), "ratings": results,
            "note": "Human sentence-support ratings, not automatic entailment or a learner study. "
                    "Report unclear/unrated counts with every rate; split compound claims if needed."}
