import argparse
from dataclasses import fields
import importlib.metadata
import json
from pathlib import Path

from .backends import BartSummarizer, ExtractiveDemo, HashEmbedder, SentenceTransformerEmbedder
from .data import Transcript
from .evaluation import evaluate_boundaries, judge_records
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
    summarize.add_argument("--backend", choices=["neural", "demo"], default="neural")
    summarize.add_argument("--device", default="cpu")
    summarize.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    summarize.add_argument("--bart-model", default="facebook/bart-large-cnn")
    summarize.add_argument("--embedding-revision")
    summarize.add_argument("--bart-revision")
    summarize.add_argument("--max-new-tokens", type=int, default=128)
    summarize.add_argument("--num-beams", type=int, default=4)
    summarize.add_argument("--threads", type=int, default=4, help="CPU torch thread count")
    defaults = Config()
    for field in fields(Config):
        default = getattr(defaults, field.name)
        kwargs = {"default": default, "type": type(default)}
        if field.name in {"chunking", "evidence"}:
            kwargs["choices"] = (["semantic", "fixed"] if field.name == "chunking" else ["mmr", "raw"])
        summarize.add_argument("--" + field.name.replace("_", "-"), **kwargs)
    evaluate = commands.add_parser("evaluate-boundaries")
    evaluate.add_argument("sedl")
    evaluate.add_argument("annotations")
    evaluate.add_argument("--tolerance-seconds", type=float, default=5.0)
    evaluate.add_argument("-o", "--output", required=True)
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
        if args.command == "summarize":
            transcript = Transcript.from_dict(read_json(args.input))
            config = Config(**{field.name: getattr(args, field.name) for field in fields(Config)})
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
            result = run(transcript, embedder, summarizer, config)
            result["generation"] = ({"max_new_tokens": args.max_new_tokens, "num_beams": args.num_beams,
                                     "do_sample": False, "seed": 0, "device": args.device,
                                     "threads": args.threads} if args.backend == "neural" else None)
            result["runtime_versions"] = {name: importlib.metadata.version(name) for name in
                                           (["numpy", "torch", "transformers", "sentence-transformers"]
                                            if args.backend == "neural" else ["numpy"])}
            write_json(args.output, result)
            print(f"Wrote {len(result['entries'])} chunks ({result['mode']}) to {args.output}")
        elif args.command == "evaluate-boundaries":
            write_json(args.output, evaluate_boundaries(read_json(args.sedl), read_json(args.annotations),
                                                       args.tolerance_seconds))
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
