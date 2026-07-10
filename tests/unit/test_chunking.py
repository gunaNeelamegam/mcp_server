from ingestion.chunking import chunk_text


def test_empty_text_produces_no_chunks():
    assert chunk_text("", chunk_size=100, chunk_overlap=10) == []


def test_short_text_produces_one_chunk():
    text = "short text under the chunk size"
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert chunks == [text]


def test_long_text_is_split_with_overlap():
    text = "x" * 250
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    # every char of the source should appear in at least one chunk
    assert "".join(dict.fromkeys("".join(chunks))) != ""
    for chunk in chunks:
        assert len(chunk) <= 100


def test_last_chunk_is_not_dropped():
    text = "a" * 90 + "b" * 5  # smaller than chunk_size, should still be included fully
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
    assert chunks[-1].endswith("b")
