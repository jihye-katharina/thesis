"""Real adapter integration with tiny RANDOM local models, not quality evaluation.

No network or pretrained artifacts required. Skip when neural extras are absent.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")
pytest.importorskip("transformers")

from sentence_transformers import SentenceTransformer, models
from tokenizers import ByteLevelBPETokenizer
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizerFast
from transformers import BertConfig, BertModel, BertTokenizerFast

from sedl.backends import BartSummarizer, SentenceTransformerEmbedder
from sedl.data import Transcript
from sedl.pipeline import Config, run


@pytest.fixture(scope="module")
def adapters(tmp_path_factory):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("random-models")
    corpus = root / "corpus.txt"
    corpus.write_text("Learning models predict labels. Evaluation uses unseen data. " * 40)
    bart_dir = root / "bart"
    bart_dir.mkdir()
    bpe = ByteLevelBPETokenizer()
    bpe.train(files=[str(corpus)], vocab_size=300,
              special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    bpe.save_model(str(bart_dir))
    tokenizer = BartTokenizerFast(vocab_file=str(bart_dir / "vocab.json"),
                                  merges_file=str(bart_dir / "merges.txt"), model_max_length=64)
    tokenizer.save_pretrained(bart_dir)
    config = BartConfig(vocab_size=len(tokenizer), d_model=16, encoder_layers=1,
                        decoder_layers=1, encoder_attention_heads=2, decoder_attention_heads=2,
                        encoder_ffn_dim=32, decoder_ffn_dim=32, max_position_embeddings=64,
                        bos_token_id=0, pad_token_id=1, eos_token_id=2,
                        decoder_start_token_id=2, forced_eos_token_id=2)
    model = BartForConditionalGeneration(config)
    # Make random generation reliably nonempty. This is a plumbing fixture only.
    with torch.no_grad():
        model.final_logits_bias[0, tokenizer.encode("a", add_special_tokens=False)[0]] = 20
    model.save_pretrained(bart_dir)

    bert_dir = root / "bert"
    bert_dir.mkdir()
    (bert_dir / "vocab.txt").write_text("\n".join([
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "learning", "models", "predict", "labels",
        "evaluation", "uses", "unseen", "data", "."]))
    tokenizer = BertTokenizerFast(vocab_file=str(bert_dir / "vocab.txt"))
    tokenizer.save_pretrained(bert_dir)
    BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                         num_attention_heads=2, intermediate_size=32,
                         max_position_embeddings=64)).save_pretrained(bert_dir)
    embedding_dir = root / "embedding"
    SentenceTransformer(modules=[models.Transformer(str(bert_dir), max_seq_length=32),
                                 models.Pooling(16)]).save(str(embedding_dir))
    return (SentenceTransformerEmbedder(str(embedding_dir)),
            BartSummarizer(str(bart_dir), max_new_tokens=8, num_beams=1))


def test_real_tokenizers_embedding_and_bart_generation(adapters):
    embedder, summarizer = adapters
    source = Transcript.from_dict({"segments": [
        {"start": 0, "end": 10, "text": "Learning models predict labels."},
        {"start": 10, "end": 20, "text": "Evaluation uses unseen data."}]})
    result = run(source, embedder, summarizer,
                 Config(min_chunk_tokens=1, max_chunk_tokens=50, evidence_tokens=30))
    assert result["entries"]
    for entry in result["entries"]:
        assert entry["summary"].strip()
        assert entry["generation_input_tokens"] <= 30
        assert entry["source_tokens"] <= 50


def test_neural_backends_refuse_overlong_inputs(adapters):
    embedder, summarizer = adapters
    with pytest.raises(ValueError, match="truncation"):
        embedder.encode(["learning " * 100])
    with pytest.raises(ValueError, match="truncation"):
        summarizer.summarize("learning " * 100)


def test_neural_generation_respects_per_chunk_lecture_allowances(adapters):
    embedder, summarizer = adapters
    source = Transcript.from_dict({"segments": [
        {"start": 0, "end": 10, "text": "Learning models predict labels."},
        {"start": 10, "end": 20, "text": "Evaluation uses unseen data."}]})
    result = run(source, embedder, summarizer,
                 Config(chunking="fixed", min_chunk_tokens=1, max_chunk_tokens=50,
                        evidence_tokens=30, summary_budget_tokens=8))
    assert result["statistics"]["allocated_output_budget"] == 8
    assert 0 < result["statistics"]["summary_tokens"] <= 8
