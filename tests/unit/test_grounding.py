from app.orchestrator.grounding import verify


def test_no_chunks_always_passes():
    assert verify("anything at all", []) is True


def test_answer_overlapping_chunk_passes():
    chunks = [{"text": "To clear customs in the UAE you need a commercial invoice and packing list."}]
    answer = "You need a commercial invoice and a packing list for UAE customs."
    assert verify(answer, chunks) is True


def test_answer_unrelated_to_chunks_fails():
    chunks = [{"text": "To clear customs in the UAE you need a commercial invoice and packing list."}]
    answer = "The weather in Antarctica is extremely cold this time of year."
    assert verify(answer, chunks) is False
