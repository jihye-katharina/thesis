"""Run five controlled, inference-only lecture experiments from a manifest."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from time import perf_counter

from .data import Transcript
from .pipeline import Config, run


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")


def variants(base, evidence_chunker):
    if evidence_chunker not in {"fixed", "sadc", "semantic"}:
        raise ValueError("evidence_chunker must be fixed, sadc, or semantic; choose on development data")
    return [(f"{chunker}_raw", replace(base, chunking=chunker, evidence="raw"))
            for chunker in ("fixed", "sadc", "semantic")] + [
        (f"{evidence_chunker}_{evidence}", replace(base, chunking=evidence_chunker, evidence=evidence))
        for evidence in ("lead", "mmr")]


def _gpu(summarizer):
    torch = getattr(summarizer, "torch", None)
    device = getattr(summarizer, "device", None)
    return (torch, device) if torch is not None and str(device).startswith("cuda") else (None, None)


def _text_hash(transcript):
    text = " ".join(" ".join(s.text.split()) for s in transcript.segments)
    return hashlib.sha256(text.encode()).hexdigest()


def run_experiment(manifest_path, output_dir, embedder, summarizer, *, generation=None):
    """No automatic tuning or quality scores; failed runs remain explicitly failed.

    Paths are relative to the manifest. Models are shared, while encoding and
    generation are repeated for each method. Model loading is excluded from run
    times. GPU memory includes the resident models and any cached allocations.
    """
    manifest_path = Path(manifest_path).resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("split") not in {"development", "test"}:
        raise ValueError("Manifest split must be development or test")
    paths = manifest.get("transcripts")
    if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
        raise ValueError("Manifest transcripts must be a nonempty list of file paths")
    resolved = [(manifest_path.parent / p).resolve() for p in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Manifest contains duplicate transcript paths")
    base = Config(**manifest.get("base_config", {}))
    base.validate(summarizer)
    conditions = variants(base, manifest.get("evidence_chunker"))
    ratio = manifest.get("summary_ratio", 0.10)
    if (isinstance(ratio, bool) or not isinstance(ratio, (float, int)) or
            not math.isfinite(ratio) or not 0 < ratio <= 1):
        raise ValueError("summary_ratio must be in (0, 1]")
    # Validate all inputs before creating the output folder.
    transcripts = [Transcript.from_dict(json.loads(p.read_text(encoding="utf-8"))) for p in resolved]
    if len({t.id for t in transcripts}) != len(transcripts):
        raise ValueError("Each transcript must have a unique video_id/id")
    if manifest["split"] == "test":
        development_path = manifest.get("development_manifest")
        if not isinstance(development_path, str):
            raise ValueError("Test manifest must reference a development_manifest to check data separation")
        dev_path = (manifest_path.parent / development_path).resolve()
        dev = json.loads(dev_path.read_text(encoding="utf-8"))
        if dev.get("split") != "development":
            raise ValueError("development_manifest must declare split development")
        dev_paths = [(dev_path.parent / p).resolve() for p in dev.get("transcripts", [])]
        if not dev_paths:
            raise ValueError("Development manifest must contain transcripts")
        dev_transcripts = [Transcript.from_dict(json.loads(p.read_text(encoding="utf-8"))) for p in dev_paths]
        if (set(resolved) & set(dev_paths) or
                {t.id for t in transcripts} & {t.id for t in dev_transcripts} or
                {_text_hash(t) for t in transcripts} & {_text_hash(t) for t in dev_transcripts}):
            raise ValueError("Development and test transcripts must be disjoint by path, ID, and normalized text")
        if (Config(**dev.get("base_config", {})) != base or
                dev.get("evidence_chunker") != manifest.get("evidence_chunker") or
                dev.get("summary_ratio", 0.1) != ratio):
            raise ValueError("Freeze base_config, evidence_chunker, and summary_ratio from development before test")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Experiment output directory must be empty; use a new directory for each run")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "manifest.json", manifest)
    if manifest["split"] == "test":
        save_json(output_dir / "development-manifest.json", dev)
    packages = ["numpy"] + ([] if embedder.name.startswith("demo/") else
                             ["torch", "transformers", "sentence-transformers"])
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "split": manifest["split"], "generation": generation,
        "models": {"embedding": embedder.name, "embedding_revision": embedder.revision,
                   "summarizer": summarizer.name, "summarizer_revision": summarizer.revision},
        "mode": "offline_demo" if embedder.name.startswith("demo/") or
                summarizer.name.startswith("demo/") else "neural",
        "runtime_versions": {p: importlib.metadata.version(p) for p in packages},
        "conditions": {name: asdict(config) for name, config in conditions},
        "measurement_note": "Single sequential pass, model loading excluded; no warm-up. "
                            "Order and cache effects can affect timings. Repeat for latency claims.",
        "runs": [],
    }
    torch, device = _gpu(summarizer)
    for lecture_index, transcript in enumerate(transcripts):
        source_tokens = summarizer.count_tokens("\n".join(s.text for s in transcript.segments))
        budget = base.summary_budget_tokens
        if budget is None:
            budget = max(1, math.floor(source_tokens * ratio))
        for name, config in conditions:
            config = replace(config, summary_budget_tokens=budget)
            row = {"transcript_id": transcript.id, "input": paths[lecture_index],
                   "condition": name, "transcript_tokens": source_tokens,
                   "output_budget_tokens": budget, "config": asdict(config)}
            if torch:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = perf_counter()
            try:
                result = run(transcript, embedder, summarizer, config)
                if torch:
                    torch.cuda.synchronize(device)
                row.update(status="completed", seconds=perf_counter() - started,
                           **result["statistics"])
                if torch:
                    row["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                    row["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
                result["experiment"] = {"split": manifest["split"], "condition": name,
                                        "manifest_sha256": report["manifest_sha256"]}
                result["runtime_versions"] = report["runtime_versions"]
                result["generation"] = generation
                result["runtime"] = {k: v for k, v in row.items()
                                     if k in {"seconds", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"}}
                filename = f"lecture-{lecture_index:03d}-{name}.json"
                save_json(output_dir / filename, result)
                row["output"] = filename
            except (ValueError, RuntimeError) as error:
                row.update(status="failed", seconds=perf_counter() - started,
                           error=f"{type(error).__name__}: {error}")
                if torch and isinstance(error, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
            report["runs"].append(row)
            save_json(output_dir / "results.json", report)
    columns = ["transcript_id", "condition", "status", "transcript_tokens", "output_budget_tokens", "chunks",
               "source_units", "source_tokens", "generation_input_tokens", "summary_tokens",
               "seconds", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "error", "output"]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["runs"])
    return report
