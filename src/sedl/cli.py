import argparse
from dataclasses import fields
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

from .backends import BartSummarizer, ExtractiveDemo, HashEmbedder, SentenceTransformerEmbedder
from .data import Transcript
from .evaluation import boundary_annotation_template, evaluate_boundaries, judge_records
from .pipeline import Config, run


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def parser():
    root = argparse.ArgumentParser(description="Explainable lecture summarisation (SEDL)")
    commands = root.add_subparsers(dest="command", required=True)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("input", help="JSON transcript: segments[{start, end, text}]")
    summarize.add_argument("-o", "--output", required=True)
    experiment = commands.add_parser("experiment", help="Run five controlled conditions")
    experiment.add_argument("--manifest", required=True)
    experiment.add_argument("--output-dir", required=True)
    for command in (summarize, experiment):
        command.add_argument("--backend", choices=["neural", "demo"], default="neural")
        command.add_argument("--device", default="cpu")
        command.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
        command.add_argument("--bart-model", default="facebook/bart-large-cnn")
        command.add_argument("--embedding-revision")
        command.add_argument("--bart-revision")
        command.add_argument("--max-new-tokens", type=int, default=128)
        command.add_argument("--num-beams", type=int, default=4)
        command.add_argument("--threads", type=int, default=4, help="CPU torch thread count")
    defaults = Config()
    for field in fields(Config):
        default = getattr(defaults, field.name)
        kwargs = {"default": default, "type": int if default is None else type(default)}
        if field.name in {"chunking", "evidence"}:
            kwargs["choices"] = (["semantic", "fixed", "sadc"] if field.name == "chunking"
                                 else ["mmr", "raw", "lead"])
        summarize.add_argument("--" + field.name.replace("_", "-"), **kwargs)
    evaluate = commands.add_parser("evaluate-boundaries")
    evaluate.add_argument("sedl")
    evaluate.add_argument("annotations")
    evaluate.add_argument("--tolerance-seconds", type=float, default=5.0)
    evaluate.add_argument("--tolerance-units", type=int, default=1)
    evaluate.add_argument("-o", "--output", required=True)
    for name in ("export-boundaries", "export-audit"):
        command = commands.add_parser(name)
        command.add_argument("sedl")
        command.add_argument("-o", "--output", required=True)
    audit = commands.add_parser("score-audit")
    audit.add_argument("csv")
    audit.add_argument("-o", "--output", required=True)
    export = commands.add_parser("export-judges")
    export.add_argument("sedl")
    export.add_argument("-o", "--output", required=True)
    judges = commands.add_parser("evaluate-judges", help="Calls an external paid judge API")
    judges.add_argument("sedl")
    judges.add_argument("--judge-model", required=True, help="Explicit OpenAI model name/version")
    judges.add_argument("--api-key-env", default="JUDGE_API_KEY")
    judges.add_argument("--faithfulness-context", choices=["source", "evidence", "both"], default="source")
    judges.add_argument("-o", "--output", required=True)
    return root


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    try:
        if args.command in {"summarize", "experiment"}:
            if args.command == "summarize":
                transcript = Transcript.from_dict(read_json(args.input))
                config = Config(**{field.name: getattr(args, field.name) for field in fields(Config)})
            else:
                read_json(args.manifest)  # Fail on a missing/invalid manifest before model downloads.
                output_dir = Path(args.output_dir)
                if output_dir.exists() and any(output_dir.iterdir()):
                    raise ValueError("Experiment output directory must be empty")
            loading_started = perf_counter()
            if args.backend == "demo":
                embedder, summarizer = HashEmbedder(), ExtractiveDemo()
            else:
                import torch
                if args.threads < 1:
                    raise ValueError("threads must be positive")
                torch.set_num_threads(args.threads)
                torch.manual_seed(0)
                embedder = SentenceTransformerEmbedder(args.embedding_model, device=args.device,
                                                       revision=args.embedding_revision)
                summarizer = BartSummarizer(args.bart_model, device=args.device,
                                            revision=args.bart_revision,
                                            max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
            generation = ({"max_new_tokens": args.max_new_tokens, "num_beams": args.num_beams,
                                     "do_sample": False, "seed": 0, "device": args.device,
                                     "threads": args.threads,
                                     "model_loading_seconds": perf_counter() - loading_started,
                                     "checkpoint_generation_config": summarizer.model.generation_config.to_dict()}
                          if args.backend == "neural" else None)
            if args.command == "experiment":
                from .experiments import run_experiment

                report = run_experiment(args.manifest, args.output_dir, embedder, summarizer,
                                        generation=generation)
                failed = sum(row["status"] == "failed" for row in report["runs"])
                print(f"Wrote {len(report['runs'])} experiment records to {args.output_dir}; {failed} failed")
                if failed:
                    arguments.exit(1, "Some runs failed; inspect results.json before comparing methods.\n")
                return
            result = run(transcript, embedder, summarizer, config)
            result["generation"] = generation
            result["runtime_versions"] = {name: importlib.metadata.version(name) for name in
                                           (["numpy", "torch", "transformers", "sentence-transformers"]
                                            if args.backend == "neural" else ["numpy"])}
            write_json(args.output, result)
            print(f"Wrote {len(result['entries'])} chunks ({result['mode']}) to {args.output}")
        elif args.command == "evaluate-boundaries":
            write_json(args.output, evaluate_boundaries(read_json(args.sedl), read_json(args.annotations),
                                                       args.tolerance_seconds, args.tolerance_units))
        elif args.command == "export-boundaries":
            write_json(args.output, boundary_annotation_template(read_json(args.sedl)))
        elif args.command == "export-audit":
            from .audit import export_audit

            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            export_audit(read_json(args.sedl), args.output)
        elif args.command == "score-audit":
            from .audit import score_audit

            write_json(args.output, score_audit(args.csv))
        elif args.command == "evaluate-judges":
            from .judges import ExternalJudge, evaluate_judges

            source = read_json(args.sedl)
            if source.get("mode") != "neural":
                raise ValueError("External research evaluation requires neural SEDL, not demo output")
            judge = ExternalJudge(args.judge_model, args.api_key_env)
            result = evaluate_judges(source, judge, args.faithfulness_context)
            result["runtime_versions"] = {name: importlib.metadata.version(name)
                                           for name in ["deepeval", "ragas", "langchain-openai"]}
            write_json(args.output, result)
        else:
            records = list(judge_records(read_json(args.sedl)))
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                      for row in records), encoding="utf-8")
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        arguments.exit(2, f"sedl: {error}\n")


if __name__ == "__main__":
    main()
