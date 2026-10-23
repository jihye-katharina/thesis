from copy import deepcopy

import pytest

from sedl.backends import ExtractiveDemo, HashEmbedder
from sedl.data import Transcript
from sedl.pipeline import run


@pytest.fixture
def whisper_input():
    return {
        "video_id": "-0JE4p3b1YE",
        "source": "whisper",
        "language": "en",
        "text": "Models learn from examples.",
        "segments": [{"start": 0.0, "end": 10.0, "text": "Models learn from examples."}],
        "provenance": {"tool": "whisper", "model": "example-model",
                       "settings": {"chunk_minutes": 10}},
    }


def test_whisper_identity_and_metadata_survive_sedl_export(whisper_input):
    original = deepcopy(whisper_input)
    transcript = Transcript.from_dict(whisper_input)
    assert transcript.id == "-0JE4p3b1YE"
    assert transcript.segments[0].id == "s00000"
    result = run(transcript, HashEmbedder(), ExtractiveDemo())
    assert result["transcript_id"] == original["video_id"]
    assert result["source_full_text"] == original["text"]
    assert result["source_metadata"] == {key: original[key]
                                          for key in ("source", "language", "provenance")}
    assert whisper_input == original


def test_metadata_has_independent_input_and_output_snapshots(whisper_input):
    transcript = Transcript.from_dict(whisper_input)
    whisper_input["provenance"]["settings"]["chunk_minutes"] = 20
    assert transcript.provenance["settings"]["chunk_minutes"] == 10
    result = run(transcript, HashEmbedder(), ExtractiveDemo())
    result["source_metadata"]["provenance"]["settings"]["chunk_minutes"] = 30
    assert transcript.provenance["settings"]["chunk_minutes"] == 10


def test_legacy_input_without_metadata_remains_supported():
    transcript = Transcript.from_dict({
        "id": "legacy-lecture", "segments": [{"start": 0, "end": 10, "text": "Hello."}],
    })
    assert transcript.id == "legacy-lecture"
    assert transcript.source is transcript.language is transcript.provenance is None


def test_video_id_takes_priority_over_legacy_id(whisper_input):
    whisper_input["id"] = "other-id"
    assert Transcript.from_dict(whisper_input).id == whisper_input["video_id"]


@pytest.mark.parametrize("key, value", [
    ("source", []), ("source", " "), ("language", 123), ("language", ""), ("provenance", []),
])
def test_invalid_metadata_rejected(whisper_input, key, value):
    whisper_input[key] = value
    with pytest.raises(ValueError, match=key):
        Transcript.from_dict(whisper_input)
