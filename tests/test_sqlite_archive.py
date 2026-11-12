import gzip
import io
import json
import tarfile

import pytest

from policy_r1.sqlite_search import build_sqlite_index, SQLiteRetriever, iter_corpus_lines


def make_tar(members):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.GNU_FORMAT) as archive:
        for name, content in members:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return output.getvalue()


@pytest.mark.parametrize("kind", ["jsonl", "gzip", "tar", "tar.gz"])
def test_content_detection_and_streaming_search(tmp_path, kind):
    payload = ('\ufeff' + json.dumps({"id": "0", "contents": "Paris — capital of France"},
                                   ensure_ascii=False) + '\n').encode("utf-8")
    if kind.startswith("tar"):
        # Even a hostile path remains a name; nothing is extracted.
        payload = make_tar([("README.txt", b"ignore me"), ("../../escaped.jsonl", payload),
                            ("nested/wiki_dump.jsonl", b'{"id":"1","contents":"Berlin Germany"}\n')])
    if kind in {"gzip", "tar.gz"}:
        payload = gzip.compress(payload)
    source = tmp_path / "wiki-18.jsonl.gz"  # Deliberately identical suffix for every format.
    source.write_bytes(payload)
    index = tmp_path / "wiki.sqlite"
    metadata = build_sqlite_index(source, index, batch_size=1)
    assert metadata["documents"] == (2 if kind.startswith("tar") else 1)
    retriever = SQLiteRetriever(index)
    try:
        assert retriever.search("Paris")[0]["contents"] == "Paris — capital of France"
    finally:
        retriever.close()
    assert not (tmp_path.parent.parent / "escaped.jsonl").exists()


def test_gnu_large_file_header_reproduces_reported_byte_124(tmp_path):
    member = tarfile.TarInfo("data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl")
    member.size = 14393573105  # Actual upstream header size; GNU uses base-256 at byte 124.
    header = member.tobuf(format=tarfile.GNU_FORMAT)
    assert header[124] == 0x80
    with pytest.raises(UnicodeDecodeError):
        header.decode("utf-8")
    source = tmp_path / "wiki-18.jsonl.gz"
    source.write_bytes(gzip.compress(header + b'{"id":"0","contents":"Paris France"}\n' + b' ' * 16384))
    lines = iter_corpus_lines(source)
    try:
        assert json.loads(next(lines))["id"] == "0"
    finally:
        lines.close()
    # Truncation after the valid first line must never publish an index.
    index = tmp_path / "wiki.sqlite"
    with pytest.raises((tarfile.ReadError, EOFError)):
        build_sqlite_index(source, index, batch_size=1)
    assert not index.exists()


def test_missing_jsonl_and_bad_gzip_trailer_do_not_publish_index(tmp_path):
    source, index = tmp_path / "wiki.gz", tmp_path / "wiki.sqlite"
    source.write_bytes(gzip.compress(make_tar([("notes.txt", b"no documents")])))
    with pytest.raises(ValueError, match="no regular .jsonl"):
        build_sqlite_index(source, index)
    assert not index.exists()
    payload = gzip.compress(make_tar([("wiki.jsonl", b'{"id":"0","contents":"Paris France"}\n')]))
    source.write_bytes(payload[:-8])
    with pytest.raises(EOFError):
        build_sqlite_index(source, index, batch_size=1)
    assert not index.exists()
    assert not list(tmp_path.glob(".wiki-*"))


def test_invalid_utf8_is_not_silently_ignored(tmp_path):
    source = tmp_path / "wiki.gz"
    source.write_bytes(gzip.compress(b'{"id":"0","contents":"broken \x80"}\n'))
    index = tmp_path / "wiki.sqlite"
    with pytest.raises(UnicodeDecodeError):
        build_sqlite_index(source, index)
    assert not index.exists()
