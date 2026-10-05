"""No-pickle wire protocol between the host and the sandboxed worker."""

from __future__ import annotations

import io
import os
import pickle
import struct
import threading
import time

import numpy as np
import pytest

from malvalid.sandbox.protocol import (
    MAGIC,
    Channel,
    ChannelClosed,
    ChannelTimeout,
    ProtocolError,
    decode_array,
    encode_array,
    pack_message,
    read_message,
    write_message,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int8, np.int64, np.uint8, np.bool_])
def test_array_round_trip(dtype) -> None:
    a = (np.arange(24).reshape(4, 6) % 2).astype(dtype)
    b = decode_array(encode_array(a))
    assert b.dtype == a.dtype and b.shape == a.shape
    np.testing.assert_array_equal(a, b)


def test_object_arrays_are_never_sent_or_decoded() -> None:
    with pytest.raises(ProtocolError, match="non-numeric"):
        encode_array(np.array([{"a": 1}], dtype=object))
    buf = io.BytesIO()
    np.save(buf, np.array([1, "x"], dtype=object), allow_pickle=True)
    with pytest.raises(ProtocolError, match="dtype"):
        decode_array(buf.getvalue())


def test_decode_rejects_oversized_and_garbage_payloads() -> None:
    with pytest.raises(ProtocolError, match="elements"):
        decode_array(encode_array(np.zeros(100)), max_elements=10)
    with pytest.raises(ProtocolError):
        decode_array(pickle.dumps(np.zeros(3)))
    with pytest.raises(ProtocolError):
        decode_array(b"\x93NUMPY\x01\x00garbage")


def test_blocking_frames_round_trip_over_a_pipe() -> None:
    r, w = os.pipe()
    try:
        x = np.random.default_rng(0).random((5, 3)).astype(np.float32)
        write_message(w, {"op": "predict_proba", "id": 3}, {"X": encode_array(x), "raw": b"MZ\x90\x00"})
        msg = read_message(r)
        assert msg.header["op"] == "predict_proba" and msg.header["id"] == 3
        np.testing.assert_array_equal(msg.array("X"), x)
        assert msg.payloads["raw"] == b"MZ\x90\x00"
        with pytest.raises(ProtocolError):
            msg.array("missing")
    finally:
        os.close(r)
        os.close(w)


def test_bad_magic_and_bounded_payloads() -> None:
    # after a bad frame the stream is out of sync (the host kills the worker), so use fresh pipes
    def frame(header: bytes, magic: bytes = MAGIC) -> bytes:
        return struct.pack(">4sI", magic, len(header)) + header

    big = b'{"ok":true,"payloads":[{"name":"y","kind":"raw","size":1000000}]}'
    frames = [
        (frame(b"{}", magic=b"EVIL"), {}, "magic"),
        (frame(big), {"max_payload": 1000}, "at most"),
        (frame(b"[1]"), {}, "not a JSON object"),
        (frame(b"{not json"), {}, "invalid frame header"),
    ]
    for data, kw, pattern in frames:
        r, w = os.pipe()
        try:
            os.write(w, data)
            with pytest.raises(ProtocolError, match=pattern):
                read_message(r, **kw)
        finally:
            os.close(r)
            os.close(w)


def test_pack_message_describes_payloads() -> None:
    segs = pack_message({"op": "x"}, {"a": encode_array(np.zeros(2)), "b": b"raw"})
    assert segs[0][:4] == MAGIC and len(segs) == 3


def test_channel_deadline_and_closed_peer() -> None:
    a_r, b_w = os.pipe()
    b_r, a_w = os.pipe()
    ch = Channel(a_r, a_w, tick=0.05)
    try:
        t0 = time.monotonic()
        with pytest.raises(ChannelTimeout):
            ch.recv(deadline=time.monotonic() + 0.3)
        assert time.monotonic() - t0 < 3

        def reply() -> None:
            m = read_message(b_r)
            write_message(b_w, {"ok": True, "id": m.header["id"]}, {"y": encode_array(np.ones(2))})

        th = threading.Thread(target=reply)
        th.start()
        ch.send({"op": "ping", "id": 1}, deadline=time.monotonic() + 5)
        got = ch.recv(deadline=time.monotonic() + 5)
        th.join()
        assert got.ok and got.array("y").tolist() == [1.0, 1.0]

        os.close(b_w)
        b_w = -1
        with pytest.raises(ChannelClosed):
            ch.recv(deadline=time.monotonic() + 5)
    finally:
        ch.close()
        for fd in (b_r, b_w):
            if fd >= 0:
                os.close(fd)


def test_tree_payload_rejects_compressed_or_inconsistent_archives() -> None:
    """The host never lets a hostile worker zip-bomb it through the tree_ensemble response."""
    import zipfile

    from malvalid.loaders.trees import from_lightgbm_dump
    from malvalid.sandbox.host import check_tree_payload
    from malvalid.testing import make_toy_corpus, train_toy_lgbm

    corpus = make_toy_corpus(n=400, seed=3)
    booster = train_toy_lgbm(corpus, rounds=5)
    good = from_lightgbm_dump(booster.dump_model()).to_bytes()
    check_tree_payload(good)  # what TreeEnsemble.to_bytes writes is accepted

    bomb = io.BytesIO()
    with zipfile.ZipFile(bomb, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("offsets.npy", b"\0" * (4 << 20))
    with pytest.raises(ValueError, match="compressed"):
        check_tree_payload(bomb.getvalue())

    odd = io.BytesIO()
    with zipfile.ZipFile(odd, "w") as zf:
        zf.writestr("evil.pkl", pickle.dumps({"a": 1}))
    with pytest.raises(ValueError, match="not a .npy"):
        check_tree_payload(odd.getvalue())
    with pytest.raises(ValueError, match="not an .npz"):
        check_tree_payload(b"garbage")
    with pytest.raises(ValueError, match="exceeds"):
        check_tree_payload(good, max_bytes=16)
