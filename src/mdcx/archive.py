# Copyright 2026 Jorge Ellena G.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The .mdcx container format.

A converted corpus, its search index and the provenance of every passage
held in a single encrypted file.

The header is stored in clear text so that the issuer, the version and the
integrity of a file can be checked without the key, which is what is needed
to decide whether to open it. The body is encrypted with AES-256-GCM, which
authenticates as well as conceals: altering one byte makes decryption fail
rather than return corrupted data. The key is derived from a passphrase with
scrypt.

Inside the body, a SQLite database with an FTS5 index provides retrieval
without any external service.

The format encrypts at rest and decrypts in memory when opened. This is not
searchable encryption, where data is queried without ever being decrypted;
that is a separate field with documented leakage attacks and per-query costs
measured in seconds.

    python -m mdcx.archive pack --output ./corpus_md --target corpus.mdcx --key "..."
    python -m mdcx.archive info corpus.mdcx
    python -m mdcx.archive search corpus.mdcx "a question" --key "..."
    python -m mdcx.archive export corpus.mdcx --target ./restored --key "..."
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import struct
import sys
import threading
import time
from pathlib import Path

from . import console

MAGIC = b"MDCX"
VERSION = 1

SCRYPT_N = 2 ** 15
SCRYPT_R = 8
SCRYPT_P = 1
KEY_BYTES = 32

# Serialises every access to a connection. SQLite is compiled in serialised mode
# here, but the prepared statement cache lives in the Python object and is not
# protected by it: two threads running the same statement can collide and return
# a wrong row without raising.
_CONNECTION_LOCK = threading.RLock()

class _Package(sqlite3.Connection):
    """A connection that carries the identity of the package it holds.

    The identity has to live on the connection rather than in a table keyed by
    its address. An address is reused as soon as the object at it is collected,
    and the caches here are never invalidated -- correctly, since a package is
    immutable once written -- so one keyed by an address hands a newly opened
    package whatever a closed one left behind. Held here it cannot outlive the
    connection, and a side table that grew for the life of the process goes with
    it.

    sqlite3.Connection is a C type: it takes no attributes and no weak
    references. A subclass declared in Python takes both, and connect() accepts
    it as a factory.
    """

    digest: str | None = None

def resolve_key(args) -> str:
    """The passphrase, from whichever way it was given.

    Four, and the order is what they cost rather than a preference. A command
    line is readable by any process on the machine, and packaging a large corpus
    takes tens of minutes: the ways that keep the secret out of the process
    table come first, and --key is the explicit way for whoever does not mind.

        MDCX_KEY        the environment, which the MCP server already reads
        --key-file      a file, whose permissions can protect it
        --key -         standard input, as tools that handle secrets do
        --key           the command line, visible while the command runs
    """
    from_file = getattr(args, "key_file", None)
    if from_file:
        return Path(from_file).read_text(encoding="utf-8").strip("\r\n")

    given = getattr(args, "key", None)
    if given == "-":
        return sys.stdin.readline().strip("\r\n")
    if given is not None:
        return given

    from_env = os.environ.get("MDCX_KEY")
    if from_env is not None:
        return from_env

    raise SystemExit(
        "no key given. Use MDCX_KEY in the environment, --key-file, --key - to "
        "read it from standard input, or --key (which is visible in the "
        "process table while the command runs)")


def _derive_key(key: str, salt: bytes) -> bytes:
    memory = 128 * SCRYPT_N * SCRYPT_R
    return hashlib.scrypt(key.encode("utf-8"), salt=salt,
                          n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=KEY_BYTES,
                          maxmem=memory * 2)

def _encrypt(data: bytes, derived_key: bytes) -> tuple[bytes, bytes]:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    nonce = os.urandom(12)
    return nonce, AESGCM(derived_key).encrypt(nonce, data, None)

def _decrypt(body: bytes, derived_key: bytes, nonce: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    return AESGCM(derived_key).decrypt(nonce, body, None)

# Provenances a date may carry, worst last. The order is what they are worth:
# a date from the source that published the work is the work's; a modification
# time is the file's, and saying so is the whole point of recording where it
# came from.
DATE_PROVENANCES = ("source", "sidecar", "front-matter", "isbn", "mtime")

# Deliberately absent: the copyright year found in the text of the document. It
# was measured and it answers badly -- a textbook reprints its front matter, so
# the year in the page is the year of the printing rather than of the edition.
# A date nobody can trust is worse than none, because none is visible.


def read_dates(path: Path) -> dict[str, tuple[str, str]]:
    """Dates supplied alongside a collection, by pseudopath or relative path.

    One record per line: ``path,date`` or ``path,date,provenance``. The third
    column is what lets whoever recovered a date from the publisher say so --
    the difference between `source` and `sidecar` is the difference between the
    work's date and one somebody typed, and only the caller knows which it is.
    """
    import csv

    dates: dict[str, tuple[str, str]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip().startswith("#") or len(row) < 2:
                continue
            key = row[0].strip()
            date = row[1].strip()
            if not key or not date:
                continue
            origin = (row[2].strip() if len(row) > 2 and row[2].strip()
                      else "sidecar")
            dates[key] = (date, origin)
    return dates


_DATE_FIELDS = ("dated:", "date:", "published:", "issued:")


def _date_of(document: dict, supplied: dict[str, tuple[str, str]],
             use_mtime: bool) -> tuple[str | None, str | None]:
    """When a work is from, from the best source that has it.

    In order: what the caller supplied, then the document's own front matter,
    then -- only if asked for -- the modification time of the file. That last
    one is the file's date and not the work's, so it is never taken unless
    requested and always says what it is.
    """
    for key in (document.get("pseudopath"), document.get("rel"),
                document.get("name")):
        if key and key in supplied:
            date, origin = supplied[key]
            return date, (origin if origin in DATE_PROVENANCES else "sidecar")

    for line in (document.get("front_matter") or "").splitlines():
        lowered = line.strip().lower()
        for field in _DATE_FIELDS:
            if lowered.startswith(field):
                value = line.split(":", 1)[1].strip().strip('"')
                if value:
                    return value, "front-matter"

    if use_mtime:
        origin_path = document.get("path")
        try:
            stamp = Path(origin_path).stat().st_mtime
        except Exception:  # noqa: BLE001
            return None, None
        import datetime

        return (datetime.datetime.fromtimestamp(
            stamp, datetime.timezone.utc).strftime("%Y-%m-%d"), "mtime")

    return None, None


def _build_database(folder: Path, semantic: bool = False,
                    reuse: dict | None = None,
                    focus: list[str] | None = None,
                    dates: dict | None = None,
                    use_mtime: bool = False) -> tuple[bytes, dict]:
    """Build the in-memory database with documents, index and provenance."""
    from . import search as B

    # A folder of Markdown, or a JSONL file with one record per line. The
    # second is for a collection that is generated rather than converted,
    # where writing it out as files and reading it back is work with nothing
    # to show for it.
    docs = (B.load_records(folder) if folder.is_file()
            else B.load_documents(folder))
    connection = sqlite3.connect(":memory:")
    connection.executescript("""
        PRAGMA journal_mode = OFF;
        CREATE TABLE document (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            pseudopath TEXT NOT NULL,
            source TEXT NOT NULL,
            folder TEXT,
            archive TEXT,
            verification_status TEXT,
            -- When the work is from, and where that was learned. Two columns
            -- rather than one, because a date without its provenance confuses
            -- "when the work was published" with "when the file was touched",
            -- and whoever reads it has no way to notice. NULL where nothing
            -- reliable was found, which is an honest answer and a different one
            -- from an invented date.
            dated TEXT,
            dated_from TEXT,
            -- Normalised text of the whole document. Literal matching runs here rather
            -- than over passages, because a quoted phrase often crosses the boundary
            -- between paragraphs.
            normalized_text TEXT
        );
        CREATE TABLE passage (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES document(id),
            position INTEGER NOT NULL,
            text TEXT NOT NULL,
            -- Searchable form of the same text. Identical to it for every
            -- script that separates words; Chinese, Japanese and Korean are
            -- split into characters so a lexical index can match them.
            search_text TEXT
        );
        -- The index is declared external to the content so the text is not stored
        -- twice: FTS5 indexes what lives in the passage table.
        CREATE VIRTUAL TABLE passage_fts USING fts5(
            search_text, content='passage', content_rowid='id', tokenize='unicode61'
        );
        -- Vector of each passage, when the package was built with semantic
        -- retrieval. Half precision: the loss against single precision is far
        -- below the differences the ranking turns on, and it halves what a
        -- corpus of many passages adds to the file.
        CREATE TABLE passage_vector (
            passage_id INTEGER PRIMARY KEY REFERENCES passage(id),
            vector BLOB NOT NULL
        );
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        -- Document frequency per term. FTS5 holds this internally but does not
        -- expose it usably, and without it the package cannot rank by the same
        -- criterion as the folder-based search.
        CREATE TABLE df (term TEXT PRIMARY KEY, passages INTEGER NOT NULL);
    """)

    supplied = dates or {}
    n_passages = 0
    for i, d in enumerate(docs, 1):
        d["dated"], d["dated_from"] = _date_of(d, supplied, use_mtime)
        text = d["text"]
        archive = ""
        status = ""
        for line in text.splitlines()[:12]:
            if line.startswith("source_format:"):
                archive = line.split(":", 1)[1].strip()
            elif line.startswith("verification_status:"):
                status = line.split(":", 1)[1].strip()
        connection.execute(
            "INSERT INTO document VALUES (?,?,?,?,?,?,?,?,?,?)",
            (i, d["name"], d["pseudopath"], d["source"], d["folder"], archive, status,
             d.get("dated"), d.get("dated_from"), d["norm"]))
        for j, block in enumerate(d["blocks"] if "blocks" in d else _split_blocks(text)):
            if not block.strip():
                continue
            n_passages += 1
            # The indexed column is never left empty. FTS5 with external content
            # reads it directly, so a null there is a passage that was not
            # indexed at all. For text that already separates words it is
            # identical to the original, and the compression of the package
            # absorbs the duplication.
            connection.execute(
                "INSERT INTO passage VALUES (?,?,?,?,?)",
                (n_passages, i, j, block,
                 B._normalize(B.segment_for_index(block))))

    connection.execute("INSERT INTO passage_fts(passage_fts) VALUES('rebuild')")

    from . import search as _B
    from collections import Counter as _Counter

    df_count: _Counter = _Counter()
    lengths: list[int] = []
    for (text,) in connection.execute("SELECT text FROM passage"):
        tk = _B.tokenize_text(_B._normalize(text))
        lengths.append(len(tk))
        for t in set(tk):
            if len(t) >= 3 or (len(t) == 1 and _B._is_cjk(t)):
                df_count[t] += 1
    connection.executemany("INSERT INTO df VALUES (?,?)", df_count.items())
    avg_length = sum(lengths) / len(lengths) if lengths else 60.0

    # The language of the corpus is recorded so that a client, a model, or the
    # query itself can tell when a question is written in another one. Retrieval
    # is lexical: a term absent from the index cannot match, and without this the
    # result is an empty answer indistinguishable from "the corpus lacks it".
    sample = " ".join(d["text"][:4000] for d in docs[:40])
    language, confidence = B.detect_language(sample)

    summary = {
        "documents": len(docs),
        "language": language,
        "language_confidence": round(confidence, 3),
        "passages": n_passages,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_folder": folder.name,
        "mean_passage_length": round(avg_length, 2),
        "indexed_terms": len(df_count),
    }
    fechados = sorted(d["dated"] for d in docs if d.get("dated"))
    if fechados:
        summary["dated_range"] = [fechados[0], fechados[-1]]
    # Reported whether or not any were found: "0 of 8 dated" is the signal
    # that the dates were lost on the way in, which is the defect this exists
    # to make visible.
    summary["dated_documents"] = [len(fechados), len(docs)]
    manifest = folder / "_manifest.json"
    if manifest.exists():
        try:
            m = json.loads(manifest.read_text(encoding="utf-8"))
            summary["conversion"] = m.get("summary", {})
        except Exception:  # noqa: BLE001
            pass
    if semantic:
        summary.update(_embed_passages(connection, reuse, focus))

    for k, v in summary.items():
        connection.execute("INSERT INTO meta VALUES (?,?)",
                    (k, json.dumps(v) if not isinstance(v, str) else v))
    connection.commit()

    data = connection.serialize()
    connection.close()
    return bytes(data), summary

def _split_blocks(text: str) -> list[str]:
    return [b for b in text.split("\n\n") if b.strip()]

def generate_signing_key() -> tuple[str, str]:
    """Create an Ed25519 key pair and return it as (private, public) hex strings.

    The private key signs packages; the public key lets anyone verify who issued
    one. Only the public half is meant to be distributed.
    """
    from cryptography.hazmat.primitives.asymmetric import ed25519

    private = ed25519.Ed25519PrivateKey.generate()
    return (private.private_bytes_raw().hex(),
            private.public_key().public_bytes_raw().hex())


def _sign(digest: str, signing_key: str) -> str:
    from cryptography.hazmat.primitives.asymmetric import ed25519

    key = ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(signing_key))
    return key.sign(digest.encode("ascii")).hex()


def verify_signature(path: Path, public_key: str) -> bool:
    """Report whether a package was signed by the holder of this public key.

    The signature covers the digest of the encrypted body, so it attests both the
    issuer and the content: altering either invalidates it. Verification needs
    neither the encryption key nor the contents of the package.
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric import ed25519

    header = read_header(path)
    signature = header.get("signature")
    if not signature:
        return False
    # The signature covers the digest recorded in the header, so it must be checked
    # together with the integrity of the body. Verifying the signature alone would
    # accept a package whose body had been replaced while the header was left intact:
    # the stored digest would still match the signature, and the content would not.
    if not header.get("_intact"):
        return False
    try:
        key = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key))
        key.verify(bytes.fromhex(signature), header["body_digest"].encode("ascii"))
    except (InvalidSignature, ValueError):
        return False
    return True


def pack(folder: Path, target: Path, key: str, issuer: str = "",
         signing_key: str = "", semantic: bool = False,
         reuse_from: Path | None = None,
         focus: list[str] | None = None,
         dates: dict | None = None, use_mtime: bool = False) -> dict:
    """Write the .mdcx file and return its figures."""
    import lzma

    if not folder.is_dir() and not folder.is_file():
        raise ValueError(f"Not a folder or a file: {folder}")

    # An empty key is not a key. scrypt derives from b"" as happily as from
    # anything else, so the whole circuit succeeded: the package was written,
    # encrypted, reported as packed, and opened again by anyone who thought to
    # try the empty string. Nothing in the output said so.
    #
    # It is caught here rather than on opening, because refusing it there would
    # make packages already written this way unreadable. What has to be stopped
    # is creating them.
    if not key or not key.strip():
        raise ValueError(
            "the key is empty: --key was given without a value, or the "
            "variable it was read from resolved to nothing. The package would "
            "be encrypted with no secret and open to anyone who tries the "
            "empty string")

    reuse = None
    if semantic and reuse_from is not None:
        from . import semantic as _S

        # Reading the previous package needs the same key, which the caller
        # already holds: a package that cannot be decrypted holds no vectors
        # that can be reused.
        reuse = reusable_vectors(Path(reuse_from), key, _S.model_name())

    t0 = time.perf_counter()
    base_score, summary = _build_database(folder, semantic=semantic, reuse=reuse,
                                          focus=focus, dates=dates,
                                          use_mtime=use_mtime)
    t_base = time.perf_counter() - t0

    # An empty package is written without complaint and fails only when queried,
    # long after the mistake. The usual cause is pointing at the folder of source
    # documents rather than at the converted Markdown.
    if not summary["documents"]:
        raise ValueError(
            f"No Markdown documents found in {folder}. "
            "This should be the output folder of a conversion, not the source documents."
        )

    t0 = time.perf_counter()
    compressed = lzma.compress(base_score, preset=6)
    t_comp = time.perf_counter() - t0

    salt = os.urandom(16)
    t0 = time.perf_counter()
    derived_key = _derive_key(key, salt)
    nonce, body = _encrypt(compressed, derived_key)
    t_encrypt = time.perf_counter() - t0

    header = {
        "file_format": "mdcx",
        "version": VERSION,
        "issuer": issuer,
        "created_utc": summary["created_utc"],
        "documents": summary["documents"],
        "passages": summary["passages"],
        "language": summary.get("language"),
        "language_confidence": summary.get("language_confidence"),
        "encryption": "AES-256-GCM",
        "key_derivation": {"algorithm": "scrypt", "n": SCRYPT_N, "r": SCRYPT_R, "p": SCRYPT_P},
        "compression": "lzma",
        "salt": salt.hex(),
        "nonce": nonce.hex(),
        "body_digest": hashlib.sha256(body).hexdigest(),
        "signature": "",
        "public_key": "",
        "conversion": summary.get("conversion", {}),
        # In the header, which is readable without the key, because it is
        # what decides whether a package is worth opening. The questions
        # themselves stay inside the encrypted body: they would say what
        # the corpus is for, which is the owner's to disclose.
        "answerable_at": summary.get("answerable_at"),
        "answerable_at_from": summary.get("answerable_at_from"),
        # The span of the collection, in the header so that it can be read
        # without the key: deciding whether a package is worth opening is
        # exactly what the header is for, and "how old is this" is part of it.
        "dated_range": summary.get("dated_range"),
        "dated_documents": summary.get("dated_documents"),
    }
    if signing_key:
        from cryptography.hazmat.primitives.asymmetric import ed25519

        private = ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(signing_key))
        header["signature"] = _sign(header["body_digest"], signing_key)
        header["public_key"] = private.public_key().public_bytes_raw().hex()

    encoded_header = json.dumps(header, ensure_ascii=False).encode("utf-8")

    with open(target, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(encoded_header)))
        f.write(encoded_header)
        f.write(body)

    return {
        "bytes_database": len(base_score),
        "bytes_compressed": len(compressed),
        "bytes_file": target.stat().st_size,
        "seconds_index": round(t_base, 2),
        "seconds_compress": round(t_comp, 2),
        "seconds_encrypt": round(t_encrypt, 2),
        **summary,
    }


def passage_digest(text: str) -> str:
    """Identity of a passage for the purpose of reusing its vector.

    The same text encoded by the same model yields the same vector, so the text
    is what identifies it. The digest is taken over the exact bytes: any edit,
    however small, produces a different passage and must be encoded again.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def reusable_vectors(path: Path, key: str, model: str) -> dict[str, bytes]:
    """Vectors from an existing package, indexed by the digest of their passage.

    Returns nothing when the package was encoded by a different model, since
    vectors from two models occupy different spaces and mixing them would
    produce a ranking over quantities that cannot be compared.
    """
    connection, _ = open_package(path, key)
    try:
        # The model is recorded in the package metadata rather than in the
        # header, which is the part readable without the key.
        row = connection.execute(
            "SELECT value FROM meta WHERE key = 'embedding_model'").fetchone()
        if not row or row[0] != model:
            return {}
        rows = connection.execute(
            "SELECT p.text, v.vector FROM passage p "
            "JOIN passage_vector v ON v.passage_id = p.id").fetchall()
        return {passage_digest(text): vector for text, vector in rows}
    finally:
        connection.close()


def _embed_passages(connection: sqlite3.Connection,
                    reuse: dict[str, bytes] | None = None,
                    focus: list[str] | None = None) -> dict:
    """Compute and store the vector of every passage, reusing what is known.

    Encoding a corpus costs far more than encoding a query, and it is done once,
    here, so that whoever receives the package pays only for their own queries.
    A passage whose text is unchanged keeps its vector, so a corpus that grows
    costs what was added rather than what it holds.
    """
    import numpy as np

    from . import semantic

    rows = connection.execute("SELECT id, text FROM passage ORDER BY id").fetchall()
    if not rows:
        return {}

    reuse = reuse or {}
    known: list[tuple[int, bytes]] = []
    to_encode: list[tuple[int, str]] = []
    for identifier, text in rows:
        vector = reuse.get(passage_digest(text))
        if vector is None:
            to_encode.append((identifier, text))
        else:
            known.append((identifier, vector))

    dimensions = 0
    if to_encode:
        vectors = np.asarray(
            semantic.encode([t for _, t in to_encode], role="passage"),
            dtype=np.float16)
        dimensions = int(vectors.shape[1])
        connection.executemany(
            "INSERT INTO passage_vector VALUES (?,?)",
            [(i, v.tobytes()) for (i, _), v in zip(to_encode, vectors)])
    if known:
        connection.executemany("INSERT INTO passage_vector VALUES (?,?)", known)
        if not dimensions:
            dimensions = len(known[0][1]) // np.dtype(np.float16).itemsize

    summary = {"embedding_model": semantic.model_name(),
               "embedding_dimensions": dimensions,
               "passages_encoded": len(to_encode),
               "passages_reused": len(known)}
    reach = _answerable_at_focus(connection, focus) if focus else None
    if reach is not None:
        # Taken from the questions themselves, so it is the threshold
        # rather than an estimate of one; the reader is told which.
        summary["answerable_at"] = reach
        summary["answerable_at_from"] = "focus"
        summary["focus"] = list(focus)
    else:
        reach = _answerable_at(connection)
        if reach is not None:
            summary["answerable_at"] = reach
            summary["answerable_at_from"] = "passages"
    return summary


# How many passages are used to measure the corpus against itself. Enough for a
# median to mean something, few enough that packing does not pay for a second
# encoding of the whole corpus.
REACH_SAMPLE = 64


def _answerable_at_focus(connection: sqlite3.Connection,
                         questions: list[str]) -> float | None:
    """The threshold taken from the questions the corpus is meant to answer.

    Calibrating against passages used as probes is an approximation, and it is
    only as good as passages resembling questions. On a corpus of catalogue
    records it is not good at all: those are all back-cover blurbs and share a
    rhetorical shape, so probes drawn from them reach each other far closer
    than any short question does. Measured, such a catalogue calibrated at
    0.7588 where a corpus of books calibrates at 0.580 -- and at 0.7588 the
    warning fires on questions the catalogue answers well, which is the false
    positive of three earlier reports, returning through the other side and for
    the same underlying reason: calibrating without questions.

    Given the questions, there is nothing to approximate. The value returned is
    the *lowest* they reach, not the median: they are all questions this corpus
    is meant to answer, so the weakest of them marks the floor of what counts
    as answered. Anything reaching as near as the worst of them deserves the
    same treatment.
    """
    import numpy as np

    from . import semantic

    wanted = [q.strip() for q in questions if q and q.strip()]
    if not wanted:
        return None
    try:
        _, matrix = _vectors(connection)
        if not len(matrix):
            return None
        asked = np.asarray(semantic.encode(wanted, role="query"),
                           dtype=np.float32)
    except Exception:  # noqa: BLE001 - a package without this is still a package
        return None

    reached = [float((matrix @ vector).max()) for vector in asked]
    return round(min(reached), 4)


def _answerable_at(connection: sqlite3.Connection) -> float | None:
    """How near this corpus comes to a question it can actually answer.

    The warning that says nothing here is about your question needs a threshold,
    and a constant cannot be one: how near a corpus comes depends on the corpus.
    Measured on packages of twelve and twenty-four passages, the same questions
    reach 0.51 and 0.55 -- a fixed cut lands inside the answered range of one and
    below the other, which is exactly how the constant has failed each time it
    was moved.

    Packing has no questions to calibrate against. What it has is passages, and
    a passage used as a query -- with the query prefix, so the asymmetry the
    model was trained with is preserved -- stands in for one: it asks something
    the corpus demonstrably contains. Its own passage is excluded, or the
    measurement would be of a text against itself.

    The earlier attempt at this compared passages as passages and produced 0.87,
    higher than any question reaches, because two passages of one book resemble
    each other more than a short question resembles either. The prefix is the
    difference between that number and this one.

    The median, not the mean: a corpus with a handful of near-duplicate passages
    would otherwise report a reach nothing else in it can attain.
    """
    import numpy as np

    from . import semantic

    rows = connection.execute(
        "SELECT id, text FROM passage ORDER BY id").fetchall()
    if len(rows) < 8:
        # Too few to say anything about the shape of the corpus.
        return None

    step = max(1, len(rows) // REACH_SAMPLE)
    chosen = [rows[i] for i in range(0, len(rows), step)][:REACH_SAMPLE]

    try:
        identifiers, matrix = _vectors(connection)
        if not len(matrix):
            return None
        probes = np.asarray(
            semantic.encode([t for _, t in chosen], role="query"),
            dtype=np.float32)
    except Exception:  # noqa: BLE001 - a package without this is still a package
        return None

    # The probe's own passage is excluded by its identifier rather than by
    # position: the two coincide today and a gap in the ids would make them
    # disagree silently, which is the kind of thing that reads as a worse corpus.
    place = {identifier: i for i, identifier in enumerate(identifiers)}
    best: list[float] = []
    for (identifier, _), probe in zip(chosen, probes):
        sims = matrix @ probe
        own = place.get(identifier)
        if own is not None:
            sims[own] = -1.0
        best.append(float(sims.max()))
    return round(float(np.median(best)), 4)


def answerable_at(connection: sqlite3.Connection) -> float | None:
    """The reach this package measured against itself when it was packed.

    None for a package made before this was measured, which is what lets the
    caller fall back rather than guess: an absent measurement is not a low one.
    """
    try:
        row = connection.execute(
            "SELECT value FROM meta WHERE key='answerable_at'").fetchone()
    except Exception:  # noqa: BLE001
        return None
    if not row:
        return None
    try:
        return float(json.loads(row[0]))
    except Exception:  # noqa: BLE001
        return None


def has_vectors(connection: sqlite3.Connection) -> bool:
    """Whether the package carries passage vectors."""
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='passage_vector'"
    ).fetchone()
    if not row:
        return False
    return connection.execute("SELECT 1 FROM passage_vector LIMIT 1").fetchone() is not None


_VECTOR_CACHE: dict[str, tuple] = {}


def _cache_key(connection: sqlite3.Connection) -> str | None:
    """What identifies the package a connection holds, or None when unknown.

    Never id(connection). An address is reused as soon as a connection is
    dropped and collected, so a cache keyed by one hands a newly opened package
    whatever a closed one left behind -- and none of these caches is ever
    invalidated, because a package is immutable once written.

    That is not hypothetical. The vector cache was keyed this way, and a package
    opened at a recycled address answered with the vectors of the package that
    had been there before. The reply looked correct: the passage text is fetched
    from this connection by passage id, so the document names and pseudopaths
    were its own. Only the ranking and the cosine belonged to another corpus,
    which is the one thing nothing on the reply shows.

    open_package records the digest of the body it decrypted on the connection
    itself. Equal digest means equal content, so it is a key that means the same
    thing for two connections onto the same package and cannot mean anything for
    two different ones.
    """
    return getattr(connection, "digest", None)


def _vectors(connection: sqlite3.Connection):
    """The matrix of passage vectors, read once per package.

    Reading and assembling them takes long enough to be felt on a corpus of many
    passages, and a server answers many queries against the same package.

    A connection this module did not open carries no digest, and is read every
    time rather than cached under a key that cannot distinguish it from another.
    """
    import numpy as np

    key = _cache_key(connection)
    if key is not None and key in _VECTOR_CACHE:
        return _VECTOR_CACHE[key]
    rows = connection.execute(
        "SELECT passage_id, vector FROM passage_vector ORDER BY passage_id").fetchall()
    identifiers = [f[0] for f in rows]
    matrix = np.frombuffer(b"".join(f[1] for f in rows), dtype=np.float16)
    matrix = matrix.reshape(len(rows), -1).astype(np.float32)
    if key is None:
        return identifiers, matrix
    _VECTOR_CACHE[key] = (identifiers, matrix)
    return _VECTOR_CACHE[key]


def semantic_query(connection: sqlite3.Connection, query_text: str,
                   limit: int = 8) -> list[dict]:
    """Rank passages by meaning rather than by word.

    This is what reaches a document written in another language: the query and
    the document share no term, and the model places them near each other
    because they say the same thing.
    """
    import numpy as np

    from . import semantic

    identifiers, matrix = _vectors(connection)
    if not identifiers:
        return []
    vector = np.asarray(semantic.encode([query_text], role="query")[0],
                        dtype=np.float32)
    scores = matrix @ vector
    best_ones = np.argsort(-scores)[:limit]

    columna = document_column(connection)
    fechas = ", d.dated, d.dated_from" if has_dates(connection) else ""
    output = []
    for position in best_ones:
        pid = identifiers[int(position)]
        row = connection.execute(
            f"SELECT d.name, d.pseudopath, d.source, p.text{fechas} "
            f"FROM passage p JOIN document d ON d.id = p.{columna} WHERE p.id = ?",
            (pid,)).fetchone()
        if row is None:
            continue
        item = {"document": row[0], "pseudopath": row[1], "source": row[2],
                "passage": row[3], "score": float(scores[int(position)]),
                "engine": "semantic"}
        if fechas:
            item["dated"], item["dated_from"] = row[4], row[5]
        output.append(item)
    return output


def language_mismatch(connection: sqlite3.Connection, query_text: str) -> str | None:
    """Explain an empty result when the query cannot match the index.

    Retrieval is lexical: a term absent from the index cannot match, however well
    the material is indexed. When none of the query terms appear in the index at
    all, and the corpus language is known and differs from what the query looks
    like, the language is almost certainly the reason, and saying so turns a
    silent zero into something the reader can act on.

    This does not make the query work. It explains why it did not.
    """
    from . import search as B

    terms = set(B.searchable_terms(B._normalize(query_text)))
    terms -= B.STOPWORDS
    if not terms:
        return None

    with _CONNECTION_LOCK:
        markers = ",".join("?" * len(terms))
        row = connection.execute(
            f"SELECT count(*) FROM df WHERE term IN ({markers})",
            tuple(terms)).fetchone()
    present = row[0] if row else 0
    if present:
        # Some term does exist in the corpus, so the empty result is about the
        # combination, not about the language.
        return None

    language = _corpus_language(connection)
    if not language:
        return ("None of the query terms appear in this corpus. If the documents "
                "are in another language, note that matching is literal and a "
                "query in a different language finds nothing.")

    query_lang, _ = B.detect_language(query_text)
    if query_lang and query_lang == language:
        return None
    detail = f" The query looks like '{query_lang}'." if query_lang else ""
    return (f"None of the query terms appear in this corpus, which is in "
            f"'{language}'.{detail} Matching is literal, so a query in another "
            f"language finds nothing even when the material is present. Try the "
            f"same question in '{language}'.")


def _corpus_language(connection: sqlite3.Connection) -> str | None:
    """Language recorded when the package was built, if any."""
    with _CONNECTION_LOCK:
        row = connection.execute(
            "SELECT value FROM meta WHERE key='language'").fetchone()
    if not row or not row[0]:
        return None
    try:
        value = json.loads(row[0])
    except Exception:  # noqa: BLE001
        value = row[0]
    return value or None


def read_header(path: Path) -> dict:
    """Header and integrity status, without requiring the key."""
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError("not an .mdcx file")
        (n,) = struct.unpack("<I", f.read(4))
        header = json.loads(f.read(n).decode("utf-8"))
        body = f.read()
    header["_intact"] = hashlib.sha256(body).hexdigest() == header.get("body_digest")
    header["_signed"] = bool(header.get("signature"))
    header["_body_bytes"] = len(body)
    return header

def open_package(path: Path, key: str) -> tuple[sqlite3.Connection, dict]:
    """Decrypt in memory and return a connection ready to query.

    Nothing is written to disk, so an open package leaves no plaintext copy."""
    import lzma

    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError("not an .mdcx file")
        (n,) = struct.unpack("<I", f.read(4))
        header = json.loads(f.read(n).decode("utf-8"))
        body = f.read()

    if hashlib.sha256(body).hexdigest() != header.get("body_digest"):
        raise ValueError("file has been altered: body digest does not match")

    derived_key = _derive_key(key, bytes.fromhex(header["salt"]))
    try:
        compressed = _decrypt(body, derived_key, bytes.fromhex(header["nonce"]))
    except Exception as exc:  # noqa: BLE001
        raise ValueError("incorrect key or corrupted file") from exc

    header["_intact"] = True
    header["_body_bytes"] = len(body)

    # check_same_thread=False is required because callers such as the MCP server
    # dispatch handlers on a thread pool, so a single cached connection is used
    # from a different thread on each call. On its own it is not safe: two threads
    # running the same statement can collide on the per-connection prepared
    # statement cache and return a wrong row without raising. Every access is
    # therefore serialised through _CONNECTION_LOCK.
    connection = sqlite3.connect(":memory:", check_same_thread=False,
                                 factory=_Package)
    connection.deserialize(lzma.decompress(compressed))
    connection.digest = header["body_digest"]
    return connection, header

_SQL_TEMPLATE = """
    SELECT d.name, d.pseudopath, d.source, p.text, bm25(passage_fts) AS score{dates}
    FROM passage_fts
    JOIN passage p ON p.id = passage_fts.rowid
    JOIN document d ON d.id = p.{document_column}
    WHERE passage_fts MATCH ?
"""

def has_dates(connection: sqlite3.Connection) -> bool:
    """Whether this package carries a date per document.

    Packages written before the columns existed do not, and asking them for one
    would fail rather than answer "unknown" -- so the query is built around what
    the package has instead of assuming a shape.
    """
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(document)")}
    except Exception:  # noqa: BLE001
        return False
    return "dated" in columns and "dated_from" in columns


def document_column(connection: sqlite3.Connection) -> str:
    """Name of the column linking a passage to its document.

    Packages written before 1.0.3 use a Spanish column name. Reading it from the
    table definition keeps those packages usable instead of failing on a name.
    """
    key = _cache_key(connection)
    cached = _COLUMN_CACHE.get(key) if key is not None else None
    if cached:
        return cached
    with _CONNECTION_LOCK:
        names = [row[1] for row in connection.execute("PRAGMA table_info(passage)")]
    name = "document_id" if "document_id" in names else "document_id"
    if key is not None:
        _COLUMN_CACHE[key] = name
    return name


def _run_match(connection: sqlite3.Connection, expr: str, limit: int,
              only: str | None) -> list[dict]:
    fechas = ", d.dated, d.dated_from" if has_dates(connection) else ""
    sql = _SQL_TEMPLATE.format(document_column=document_column(connection),
                               dates=fechas)
    params: list = [expr]
    if only:
        sql += " AND d.source = ?"
        params.append(only.upper())
    sql += " ORDER BY score LIMIT ?"
    params.append(limit)
    try:
        with _CONNECTION_LOCK:
            rows = connection.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []
    out = []
    for r in rows:
        item = {"document": r[0], "pseudopath": r[1], "source": r[2],
                "passage": r[3], "score": round(-r[4], 3)}
        if fechas:
            item["dated"], item["dated_from"] = r[5], r[6]
        out.append(item)
    return out

# How much a date is allowed to weigh against what the question is about.
#
# Small on purpose, and declared rather than implicit. Recency is a
# preference of whoever asks, not a property of the corpus: a work from 1970
# can be the right answer, and in mathematics it often is. At equal footing a
# third ranking would decide a third of the outcome, which would let the date
# overrule the subject.
#
# What that buys, exactly, because the arithmetic of reciprocal rank is not
# obvious: where the two engines disagree about which passage comes first, the
# date decides between them -- measured, a work of 2025 rises above one of 2015
# and the older one stays in the list. Where both engines put the same passage
# first, no small weight moves it: with k=60 the gap between consecutive places
# is 2*(1/61 - 1/62), and outweighing it would take a weight above 2, which is
# more than either engine carries.
#
# That boundary is the feature rather than a limitation of it. A preference
# should reorder what relevance considers comparable and should not be able to
# overrule what both engines agree answers better.
RECENCY_WEIGHT = 0.25

K1 = 1.5
B_LENGTH = 0.45
DOC_TOP_PASSAGES = 8

CANDIDATES = 1200

def lexical_query(connection: sqlite3.Connection, query_text: str, limit: int = 8,
              only: str | None = None) -> list[dict]:
    """Resolve a query by word, ranking by document rather than by isolated passage."""
    from . import search as B

    phrase = query_text.strip().split(".")[0][:160].strip()
    effective = phrase if len(phrase.split()) >= 5 else query_text

    terms = B.searchable_terms(B._normalize(effective))
    terms = B.expand_terms(terms, _corpus_language(connection))
    if not terms:
        return []
    distinct_terms = set(terms)

    expr = " OR ".join(f'"{B.segment_for_index(t)}"' for t in distinct_terms)
    candidates = _run_match(connection, expr, CANDIDATES, only)
    if not candidates:
        return []

    df, n_passages, avg_length = _corpus_statistics(connection)

    by_document: dict[str, list[dict]] = {}
    for r in candidates:
        freq = _term_frequencies(r["passage"], distinct_terms)
        if not freq:
            continue
        length = max(len(B.tokenize_text(B._normalize(r["passage"]))), 1)
        score = 0.0
        for t, f in freq.items():
            d_t = df.get(t, 1)
            idf = math.log(1 + (n_passages - d_t + 0.5) / (d_t + 0.5))
            score += idf * (f * (K1 + 1)) / (
                f + K1 * (1 - B_LENGTH + B_LENGTH * length / avg_length))
        r = dict(r)
        r["score"] = round(score, 3)
        r["terms"] = sorted(freq)
        by_document.setdefault(r["document"], []).append(r)

    ranking = []
    for name, passages in by_document.items():
        passages.sort(key=lambda x: -x["score"])
        base_score = sum(x["score"] for x in passages[:DOC_TOP_PASSAGES])
        coverage = max(len(x["terms"]) for x in passages) / max(len(distinct_terms), 1)
        ranking.append((base_score * coverage, passages))
    ranking.sort(key=lambda pair: -pair[0])

    if len(phrase.split()) >= 5:
        needle = B._normalize(phrase)
        with _CONNECTION_LOCK:
            preferred = [n for (n, t) in connection.execute(
                "SELECT name, normalized_text FROM document") if t and needle in t]
        if preferred:
            position = {d: i for i, d in enumerate(preferred)}
            ranking.sort(key=lambda pair: (position.get(pair[1][0]["document"], len(position)),
                                          -pair[0]))

    out: list[dict] = []
    for round_index in range(DOC_TOP_PASSAGES):
        for score, passages in ranking:
            if round_index < len(passages):
                r = dict(passages[round_index])
                r["document_score"] = round(score, 3)
                out.append(r)
                if len(out) >= limit:
                    return out
    return out[:limit]


def query(connection: sqlite3.Connection, query_text: str, limit: int = 8,
          only: str | None = None, mode: str = "auto",
          prefer: str | None = None, notes: dict | None = None) -> list[dict]:
    """Resolve a query, by word and by meaning where the package allows it.

    The two engines answer different questions. The lexical one finds documents
    that contain the words, which is exact and is what serves a query in the
    language the documents are written in. The dense one finds documents that
    mean the same thing, which is what reaches a document written in another
    language, where no word is shared.

    Neither replaces the other: used alone, the dense engine loses precision on
    the language of the query, and the lexical one cannot leave that language at
    all. They are merged by reciprocal rank, which needs no common scale between
    a BM25 score and a cosine similarity.

    The mode selects the engines. Left at auto, meaning is used when the package
    carries vectors and the dependency is installed, and the query falls back to
    words alone whenever it is not, without failing.

    A preference can be asked for and be impossible to honour: it orders the
    fusion of two engines, so there has to be a fusion, and it orders by date, so
    something in the answer has to carry one. Neither is an error and neither
    changes the answer, but from outside they are indistinguishable from a
    preference that applied and moved nothing. Pass a dict as `notes` to be told
    which happened: it comes back with `prefer_applied`, and with
    `prefer_reason` when the answer is no.
    """
    if mode not in ("auto", "lexical", "semantic"):
        raise ValueError(f"mode must be auto, lexical or semantic, not {mode!r}")
    if prefer not in (None, "recent"):
        raise ValueError(f"prefer must be None or 'recent', not {prefer!r}")

    def note(applied: bool, why: str = "") -> None:
        """Say whether the preference was honoured, when one was asked for.

        Silent when none was asked for: a caller that never mentioned `prefer`
        should not have to read about it.
        """
        if notes is None or not prefer:
            return
        notes["prefer_applied"] = applied
        if applied:
            notes.pop("prefer_reason", None)
        else:
            notes["prefer_reason"] = why

    # The direction is stored uppercase and arrives lowercase: the CLI declares
    # its choices that way and the MCP server lowercases whatever it is handed.
    # The lexical engine folded it and the dense one compared it raw, so a
    # restricted query silently discarded every dense result -- and with it the
    # only engine that reaches a document written in another language. It looked
    # like a corpus with nothing to say rather than a filter that matched
    # nothing. Folding it once here leaves both engines comparing the same form.
    only = only.upper() if only else None

    # Every early return below leaves a preference unapplied, and each for a
    # different reason. Saying which one is the whole point of `notes`: from
    # outside, a preference that could not be applied looks exactly like one
    # that applied and moved nothing.
    lexical = [] if mode == "semantic" else lexical_query(
        connection, query_text, limit * 3, only)
    if mode == "lexical":
        note(False, "the query ran on words alone, and the preference orders "
                    "the fusion of both engines")
        return lexical[:limit]

    if not _semantic_ready(connection):
        note(False, "the package carries no meaning index, so there is no "
                    "fusion to order"
             if not has_vectors(connection) else
             "retrieval by meaning is not installed here: "
             "pip install 'mdcx[multilingual]'")
        if mode == "semantic":
            return []
        return lexical[:limit]

    dense = semantic_query(connection, query_text, limit * 3)
    if only:
        dense = [r for r in dense if r.get("source") == only]
    if mode == "semantic":
        note(False, "the query ran on meaning alone, and the preference orders "
                    "the fusion of both engines")
        return dense[:limit]
    if not dense or not lexical:
        note(False, "only one engine answered, and the preference orders "
                    "the fusion of both")
        return (lexical or dense)[:limit]

    from . import semantic as S

    def key(r: dict) -> tuple:
        return (r["document"], r["passage"][:120])

    by_key = {}
    for r in lexical + dense:
        by_key.setdefault(key(r), r)
    rankings = [[key(r) for r in lexical], [key(r) for r in dense]]
    weights: tuple[float, ...] | None = None
    if prefer == "recent":
        # A third ranking, by date, rather than a decay multiplying the
        # cosine: weighting scores that share no scale reintroduces exactly
        # the arbitrariness that fusing by rank avoids. An order by date is a
        # legitimate ranking and fuses like the others.
        #
        # Undated passages keep their place instead of sinking. The absence
        # of a date says nothing about the age of the work, and treating it
        # as old would let a gap in the metadata decide the answer.
        dated = [r for r in by_key.values() if r.get("dated")]
        if dated:
            newest = sorted(dated, key=lambda r: str(r["dated"]), reverse=True)
            rankings.append([key(r) for r in newest])
            weights = (1.0, 1.0, RECENCY_WEIGHT)
            note(True)
        else:
            note(False, _why_no_dates(connection))
    order = S.fuse(rankings, weights=weights)
    return [by_key[k] for k in order[:limit]]


def _why_no_dates(connection: sqlite3.Connection) -> str:
    """Which of the three ways an answer ends up with no date to order by.

    They are worth telling apart because each is undone differently, and only
    the third is about this particular query:

    - the package predates dates entirely, and has no column to hold one;
    - it has the column and nothing in it, which is what packing without
      `--dates` leaves behind on a corpus whose documents carry no front matter;
    - it is dated, and this answer happens to have drawn passages from the
      documents that are not.

    Answering all three with one sentence would send someone repacking a corpus
    that is already dated.
    """
    if not has_dates(connection):
        return ("this package records no dates: it was written before mdcx "
                "stored them")
    dated = connection.execute(
        "SELECT 1 FROM document WHERE dated IS NOT NULL LIMIT 1").fetchone()
    if not dated:
        return ("no document in this package is dated: pack it again with "
                "--dates to supply them")
    return "nothing in this answer carries a date, though the package has some"


def _semantic_ready(connection: sqlite3.Connection) -> bool:
    """Whether this package and this interpreter can retrieve by meaning."""
    try:
        from . import semantic as S
    except ImportError:
        return False
    if not has_vectors(connection):
        return False
    if not S.available():
        return False
    # A package encodes its passages with one model, and a query encoded with a
    # different one lands somewhere else in the space. Comparing the two returns
    # confident nonsense, so the mismatch disables meaning rather than reporting
    # results that look ranked.
    expected = connection.execute(
        "SELECT value FROM meta WHERE key = 'embedding_model'").fetchone()
    return bool(expected) and expected[0] == S.model_name()


def _term_frequencies(text: str, terms: set[str]) -> dict[str, int]:
    from . import search as B

    count: dict[str, int] = {}
    for t in B.tokenize_text(B._normalize(text)):
        if t in terms:
            count[t] = count.get(t, 0) + 1
    return count

_STATS_CACHE: dict = {}

# Resolved column name per package, so the lookup happens once.
_COLUMN_CACHE: dict = {}

def _corpus_statistics(connection: sqlite3.Connection) -> tuple[dict, int, float]:
    """Document frequency per term and mean passage length, as packed.

    These feed BM25 directly, so reading them from another package does not
    fail: it reweights every term against a corpus the passages were not in.
    """
    key = _cache_key(connection)
    if key is not None and key in _STATS_CACHE:
        return _STATS_CACHE[key]
    with _CONNECTION_LOCK:
        df = {t: n for t, n in connection.execute("SELECT term, passages FROM df")}
        row = connection.execute(
            "SELECT value FROM meta WHERE key='passages'").fetchone()
    n = int(json.loads(row[0])) if row else max(len(df), 1)
    row = connection.execute(
        "SELECT value FROM meta WHERE key='mean_passage_length'").fetchone()
    lm = float(json.loads(row[0])) if row else 60.0
    if key is None:
        return df, n, lm
    _STATS_CACHE[key] = (df, n, lm)
    return _STATS_CACHE[key]

def export(path: Path, key: str, target: Path) -> dict:
    """Rebuild the Markdown folder from the package.

    Directory structure is restored from the pseudopath stored with each document."""
    connection, header = open_package(path, key)
    try:
        rows = connection.execute(
            "SELECT d.pseudopath, d.name, group_concat(p.text, char(10) || char(10)) "
            "FROM document d JOIN passage p ON p." + document_column(connection) + " = d.id "
            "GROUP BY d.id ORDER BY p.position").fetchall()
        written = 0
        for pseudopath, name, text in rows:
            relative = pseudopath[2:] if pseudopath.startswith("@/") else pseudopath
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text((text or "") + "\n", encoding="utf-8")
            written += 1
    finally:
        connection.close()
    return {"documents": written, "target": str(target),
            "created_utc": header.get("created_utc")}

def _direction(value: str) -> str:
    """Human-readable direction label."""
    return {"SENT": "SENT", "RECEIVED": "RECEIVED",
            "EMITIDO": "SENT", "RECIBIDO": "RECEIVED"}.get(value, "OTHER")


def main() -> int:
    console.configure()

    ap = argparse.ArgumentParser(description="The .mdcx format: an indexed, encrypted, portable corpus")
    # The version, from the installed metadata. SECURITY.md asks a reporter to
    # quote it, and until now the command it names for that errored out. The
    # import is local because __init__ imports this module, and a module-level
    # import would close the circle.
    from . import __version__
    ap.add_argument("--version", action="version", version=f"mdcx {__version__}")
    sub = ap.add_subparsers(dest="action", required=True)

    def key_argument(parser) -> None:
        """The ways a key may arrive, and what each one costs.

        --key is kept because removing it would break everyone, and for a test
        package on one's own machine it is perfectly reasonable. What it cannot
        be is the only way: a command line is readable by any process on the
        machine -- /proc/<pid>/cmdline on Linux, equivalents elsewhere -- and
        packaging a large corpus takes tens of minutes, during which the secret
        that protects it is in the process table for anyone to read. It was
        found that way: a ps to check on progress returned the key.
        """
        parser.add_argument(
            "--key", default=None,
            help="the passphrase. Visible in the process table while the "
                 "command runs; prefer MDCX_KEY, --key-file or --key - for "
                 "anything that matters")
        parser.add_argument(
            "--key-file", metavar="PATH", default=None,
            help="read the passphrase from this file, whose permissions can "
                 "protect it")

    e = sub.add_parser("pack")
    e.add_argument("--output", default="Output",
                   help="folder of Markdown to pack, or a .jsonl file with "
                        "one record per line, each with name and text")
    e.add_argument("--target", default="corpus.mdcx")
    key_argument(e)
    e.add_argument("--issuer", default="")
    e.add_argument("--signing-key", default="",
                   help="hex private key to sign the package with")
    e.add_argument("--dates", metavar="FILE",
                   help="a CSV of path,date[,provenance] giving when each "
                        "work is from. The third column is how a date "
                        "recovered from the publisher says so: without it a "
                        "reader cannot tell the work's date from one somebody "
                        "typed")
    e.add_argument("--date-from-mtime", action="store_true",
                   help="fall back to the file's modification time, recorded "
                        "as such. It is the file's date and not the work's, so "
                        "it is never used unless asked for")
    e.add_argument("--focus", action="append", metavar="QUESTION",
                   help="a question this package is meant to answer. Given "
                        "one or more, the threshold for 'nothing here is "
                        "about that' is taken from them instead of being "
                        "estimated from passages, which a corpus of "
                        "similarly shaped records estimates badly. Repeatable.")
    e.add_argument("--reuse", metavar="PACKAGE",
                   help="reuse the vectors of an existing package for passages "
                        "whose text is unchanged, so that packaging costs what "
                        "was added rather than what the corpus holds. Requires "
                        "the same key and the same model.")
    e.add_argument("--multilingual", action="store_true",
                   help="also index meaning, so that a query in one language "
                        "reaches documents written in another. Needs the "
                        "multilingual extra and encodes the whole corpus once.")

    k = sub.add_parser("keygen")

    v = sub.add_parser("verify")
    v.add_argument("path")
    v.add_argument("--public-key", required=True)

    i = sub.add_parser("info")
    i.add_argument("path")

    x = sub.add_parser("export")
    x.add_argument("path")
    x.add_argument("--target", required=True)
    key_argument(x)

    b = sub.add_parser("search")
    b.add_argument("path")
    b.add_argument("query_text")
    key_argument(b)
    b.add_argument("--limit", type=int, default=5)
    b.add_argument("--only", choices=["received", "sent"])
    b.add_argument("--mode", choices=["auto", "lexical", "semantic"], default="auto",
                   help="which engines answer: words, meaning, or both")
    b.add_argument("--prefer", choices=["recent"],
                   help="order comparable answers newest first. It orders "
                        "rather than filters: an older work that answers "
                        "better still comes back")

    args = ap.parse_args()

    if args.action == "pack":
        r = pack(Path(args.output), Path(args.target), resolve_key(args), args.issuer,
                 args.signing_key, semantic=args.multilingual,
                 reuse_from=Path(args.reuse) if args.reuse else None,
                 focus=args.focus,
                 dates=read_dates(Path(args.dates)) if args.dates else None,
                 use_mtime=args.date_from_mtime)
        print(f"Packed: {args.target}")
        print(f"  documents {r['documents']}   passages {r['passages']}")
        print(f"  database {r['bytes_database']:,} -> compressed {r['bytes_compressed']:,} "
              f"-> file {r['bytes_file']:,} bytes".replace(",", "."))
        print(f"  index {r['seconds_index']}s  compress {r['seconds_compress']}s  "
              f"encrypt {r['seconds_encrypt']}s")
        if r.get("embedding_model"):
            print(f"  meaning indexed with {r['embedding_model']} "
                  f"({r['embedding_dimensions']} dimensions)")
            if r.get("passages_reused"):
                print(f"  passages encoded {r['passages_encoded']:,}   "
                      f"reused {r['passages_reused']:,}".replace(",", "."))
        return 0

    if args.action == "keygen":
        private, public = generate_signing_key()
        print("Keep the private key secret; distribute only the public key.")
        print(f"  private: {private}")
        print(f"  public : {public}")
        return 0

    if args.action == "verify":
        valid = verify_signature(Path(args.path), args.public_key)
        print("Signature valid: the package was issued by the holder of this key "
              "and has not been altered." if valid else
              "Signature not valid: unsigned package, different key, or altered content.")
        return 0 if valid else 1

    if args.action == "info":
        header = read_header(Path(args.path))
        print(f"Format    : {header['file_format']} v{header['version']}")
        print(f"Issuer    : {header.get('issuer') or '(not declared)'}")
        print(f"Created   : {header['created_utc']}")
        print(f"Content   : {header['documents']} documents, {header['passages']} passages")
        print(f"Encryption: {header['encryption']} with {header['key_derivation']['algorithm']}")
        print(f"Integrity : {'intact' if header['_intact'] else 'ALTERED'}")
        print(f"Signature : {'present' if header.get('_signed') else 'none'}"
              + (f" (public key {header['public_key'][:16]}...)" if header.get('public_key') else ""))
        fechados = header.get("dated_documents")
        if fechados:
            span = header.get("dated_range")
            print(f"Dated     : {fechados[0]} of {fechados[1]} documents"
                  + (f", {span[0]} to {span[1]}" if span else
                     " (none carries a date)"))
        if header.get("answerable_at"):
            origin = header.get("answerable_at_from") or "passages"
            print(f"Answerable: {header['answerable_at']} "
                  + ("(taken from the questions the package was given)"
                     if origin == "focus"
                     else "(estimated from its own passages)"))
        if header.get("conversion"):
            print(f"Conversion: {json.dumps(header['conversion'], ensure_ascii=False)[:200]}")
        return 0

    if args.action == "export":
        r = export(Path(args.path), resolve_key(args), Path(args.target))
        print(f"Exported {r['documents']} documents to {r['target']}")
        return 0

    connection, header = open_package(Path(args.path), resolve_key(args))
    notes: dict = {}
    results = query(connection, args.query_text, args.limit, args.only, args.mode,
                    prefer=args.prefer, notes=notes)
    print(f"{len(results)} passage(s)")
    # Only when a preference was asked for and could not be honoured. Saying
    # nothing when it worked keeps the line meaningful: an unchanged order with
    # no line under it means the preference ran and found nothing to move.
    if notes.get("prefer_applied") is False:
        print(f"--prefer {args.prefer} was NOT applied: {notes['prefer_reason']}")
    print()
    # The position, not the score. The list is merged by reciprocal rank, and
    # the numbers behind it share no scale: a BM25 score is unbounded and
    # depends on the corpus it was measured in, a cosine runs from zero to one
    # and does not. Printed in one column they invited exactly the comparison
    # they cannot support -- 3.589 above 0.344 reads as far better and means
    # nothing of the sort. The MCP reply dropped the score for this reason;
    # this surface went on printing it.
    for position, r in enumerate(results, start=1):
        print("-" * 96)
        print(f"{position}. [{r['source']}] {r['document']}")
        print(f"{r['pseudopath']}")
        print(r["passage"][:1200])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
