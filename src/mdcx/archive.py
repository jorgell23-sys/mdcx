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
import contextlib
import hashlib
import json
import math
import os
import shutil
import sqlite3
import struct
import sys
import tempfile
import threading
import time
from pathlib import Path

from . import console
from . import shapekey

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


# How much of the database is held at a time while it is being sealed.
#
# The whole of it used to be, three times over: the serialised database, its
# compressed form and the encrypted result, all alive together and nothing
# written until all three existed. Measured by a consumer on 134,677 documents
# and 4.17 GiB of text: 35.6 GB of resident memory after 34 minutes, growing
# about 5 GB a minute, and no package written. There was no size at which it
# stopped being reasonable and started being fatal -- it simply grew until the
# machine ran out, with no error, no warning and no parameter to ask for less.
SEAL_BLOCK_BYTES = 8 * 1024 * 1024

# How many token counts to write at once. Large enough that the round trips do
# not dominate, small enough that the pending list is not a function of the
# corpus.
_TOKENS_BATCH = 20_000


# What may compress a body, and what the header then says it was compressed
# with. The header has always carried the name; until now it could only ever
# say one thing.
#
# Measured by a consumer on 784 MB of their own database: LZMA wrote 254 MB and
# took 14.01 s to read back, zstd at level 3 wrote 233 MB and took 1.14 s. On
# that material zstd is smaller *and* faster to open, so this is not a trade
# between size and speed -- which is why it is worth offering rather than
# arguing about. Opening is what a server pays before it can answer anything.
#
# LZMA stays the default: it is what every package written so far says, and a
# package is opened by what its header names.
COMPRESSORS = ("lzma", "zstd")

DEFAULT_COMPRESSION = "lzma"

# What level to use when zstd is asked for without one. Level 3 is zstd's own
# default and the level the measurement above was taken at.
ZSTD_LEVEL = 3


def _zstd():
    """The zstd module, or a refusal that says how to get it."""
    try:
        import zstandard
    except ImportError as missing:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "This package is compressed with zstd, which needs the `zstandard` "
            "module: pip install zstandard") from missing
    return zstandard


def _decompress(body: bytes, compression: str) -> bytes:
    """Undo whatever compressed this body, by name.

    A package written before this existed carries no name, and it was LZMA.
    """
    name = (compression or DEFAULT_COMPRESSION).lower()
    if name == "zstd":
        return _zstd().ZstdDecompressor().stream_reader(body).read()
    if name in ("lzma", "xz"):
        import lzma

        return lzma.decompress(body)
    raise ValueError(
        f"This package says it was compressed with {compression!r}, which this "
        "version does not know how to read.")


def _seal(source: Path, target: Path, derived_key: bytes,
          preset: int, compression: str = DEFAULT_COMPRESSION) -> tuple[bytes, str, int]:
    """Compress, encrypt and hash a file into another, a block at a time.

    Returns the nonce, the digest of what was written, and how many bytes that
    was -- the three things the header needs and the only three that have to
    outlive the operation.

    The bytes produced are the same ones `lzma.compress` and `AESGCM.encrypt`
    produce for the same input: XZ is a stream format and GCM is a stream
    cipher whose tag goes at the end, which is exactly what `AESGCM.encrypt`
    appends. So this changes what the packing costs and not what a package is,
    and a reader of any version opens it.
    """
    import lzma

    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    nonce = os.urandom(12)
    encryptor = Cipher(algorithms.AES(derived_key), modes.GCM(nonce)).encryptor()
    if compression == "zstd":
        # Its own level scale, not LZMA's. `preset` is passed through where a
        # caller named one, since both run 0..9 in the same direction, and the
        # zstd default otherwise.
        compressor = _zstd().ZstdCompressor(level=preset or ZSTD_LEVEL).compressobj()
    else:
        compressor = lzma.LZMACompressor(preset=preset)
    digest = hashlib.sha256()
    written = 0

    def put(chunk: bytes) -> None:
        nonlocal written
        if not chunk:
            return
        sealed = encryptor.update(chunk)
        if sealed:
            digest.update(sealed)
            out.write(sealed)
            written += len(sealed)

    with open(source, "rb") as raw, open(target, "wb") as out:
        while True:
            block = raw.read(SEAL_BLOCK_BYTES)
            if not block:
                break
            put(compressor.compress(block))
        put(compressor.flush())
        put(encryptor.finalize())
        # The authentication tag, which `AESGCM.encrypt` puts at the end of the
        # ciphertext and `AESGCM.decrypt` expects to find there.
        digest.update(encryptor.tag)
        out.write(encryptor.tag)
        written += len(encryptor.tag)

    return nonce, digest.hexdigest(), written

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


# How a package answers a quotation that runs past the end of a passage.
#
# "document" keeps the whole normalised text of every document and looks in it.
# There is no limit to how long a quotation may be, and it costs a second copy
# of the corpus -- measured on one package, 257.4 MB beside the 257.3 MB of the
# passages, a third of the file.
#
# "boundary" indexes the join instead: the tail of each passage against the head
# of the next, in a contentless FTS5 table that keeps the index and not the
# text. A quotation that straddles one cut is found by a phrase match, which is
# what the whole copy was being read for. What it cannot do is find one longer
# than the join, which the whole copy can -- so this is a choice, not a
# replacement.
QUOTE_STRATEGIES = ("document", "boundary")

# How much of each side of a cut the join carries. 250 characters is about forty
# words, so a quotation of some eighty words spanning one cut fits whole.
EDGE_CHARACTERS = 250


def _build_edges(connection: sqlite3.Connection) -> int:
    """Index the join between each passage and the next of its document.

    Contentless, because the joined text is not a passage and must never be
    returned as one: what a match gives back is the rowid, which is the id of
    the passage that *opens* the join and does exist in the corpus.
    """
    from . import search as B

    connection.execute(
        "CREATE VIRTUAL TABLE passage_edge_fts USING fts5(txt, content='', "
        "tokenize=\"unicode61 categories 'L* N* Co Mn Mc'\")")
    rows = connection.execute(
        "SELECT id, text, LEAD(text) OVER (PARTITION BY document_id "
        "ORDER BY position) FROM passage").fetchall()
    joins = [(identifier,
              B._normalize(head[-EDGE_CHARACTERS:] + " " + tail[:EDGE_CHARACTERS]))
             for identifier, head, tail in rows if tail]
    connection.executemany(
        "INSERT INTO passage_edge_fts(rowid, txt) VALUES (?,?)", joins)
    return len(joins)


def _has_edges(connection: sqlite3.Connection) -> bool:
    """Whether this package indexed the joins between passages."""
    try:
        with _CONNECTION_LOCK:
            return bool(connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='passage_edge_fts'").fetchone())
    except Exception:  # noqa: BLE001
        return False


def _build_database(folder: Path, into: Path, semantic: bool = False,
                    reuse: dict | None = None,
                    focus: list[str] | None = None,
                    dates: dict | None = None,
                    use_mtime: bool = False,
                    shapes: bool = False,
                    quotes: str = "document",
                    expressions: bool = False) -> tuple[int, dict]:
    """Build the database with documents, index and provenance, into a file.

    Written out rather than handed back as bytes. `serialize()` returns the
    whole database as one object, which was then compressed into a second and
    encrypted into a third -- three copies of the corpus alive at once for a
    value nothing reads in between. Writing it to a file leaves the pages where
    SQLite already put them and lets the sealing read them a block at a time.

    Returns how many bytes were written, which is the one thing about it the
    caller reports.
    """
    from . import search as B

    # A folder of Markdown, or a JSONL file with one record per line. The
    # second is for a collection that is generated rather than converted,
    # where writing it out as files and reading it back is work with nothing
    # to show for it.
    # How long each part of indexing took, so that the one number a caller sees
    # can be accounted for. `seconds_index` on its own says a corpus took six
    # hours without saying whether the answer is a different corpus, a different
    # batch, or nothing the caller can do -- and a consumer who profiled it
    # attributed the whole of it to the model, which turned out to be a small
    # part of it.
    spent: dict[str, float] = {}
    clock = time.perf_counter

    @contextlib.contextmanager
    def phase(name: str):
        started = clock()
        try:
            yield
        finally:
            spent[name] = round(spent.get(name, 0.0) + clock() - started, 3)

    def timed(source):
        """The same documents, with the time spent producing them recorded.

        Reading is no longer a stage that finishes before the next one starts,
        so it can no longer be timed by wrapping a call. Without this the
        breakdown would report reading at nought and quietly charge it to
        inserting, which is worse than not reporting it.
        """
        while True:
            started = clock()
            try:
                document = next(source)
            except StopIteration:
                spent["read"] = round(spent.get("read", 0.0) + clock() - started, 3)
                return
            spent["read"] = round(spent.get("read", 0.0) + clock() - started, 3)
            yield document

    docs = timed(iter(B.load_records(folder) if folder.is_file()
                      else B.iter_documents(folder)))
    # Built in the file rather than in memory.
    #
    # An in-memory database is the corpus again, with its index on top, and it
    # is invisible to a Python memory profile because it is C -- which is why
    # the first measurements of this looked better than the machine did.
    # Nothing reads it between building and writing, so building it where it is
    # going to be written removes the copy rather than moving it.
    into.unlink(missing_ok=True)
    connection = sqlite3.connect(into)
    connection.executescript("""
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
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
            -- Normalised text of the whole document. Literal matching runs here
            -- rather than over passages, because a quoted phrase often crosses
            -- the boundary between paragraphs.
            --
            -- It is a second copy of the corpus, and on a large collection that
            -- is what it costs: measured on one package, 257.4 MB against the
            -- 257.3 MB of the passages themselves, a third of the file. NULL
            -- where the package indexes the boundaries instead -- see
            -- `passage_edge_fts` and the `quotes` argument to `pack`.
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
            search_text TEXT,
            -- How many tokens this passage holds, by the same rule that built
            -- `df`. BM25 needs it to normalise, and a query used to get it by
            -- tokenising every candidate passage again -- measured at half the
            -- time a lexical query took. It is known here, where the passages
            -- are being counted anyway, so it is written down once instead of
            -- recomputed on every query for the life of the package.
            tokens INTEGER
        );
        -- The index is declared external to the content so the text is not stored
        -- twice: FTS5 indexes what lives in the passage table.
        --
        -- The categories are named because the default set -- letters, numbers
        -- and private use -- leaves out the combining marks, and a script that
        -- writes its vowels as marks is therefore cut at every one of them.
        -- Measured on one Hindi passage: 21 terms in the index against the 13
        -- words it holds, so a word was indexed as fragments and the term a
        -- reader would search for was not among them.
        --
        -- That also made the index and this module disagree about what a
        -- passage contains, which is what a lexical score is computed from.
        -- With Mn and Mc included the two produce the same 13 terms.
        --
        -- The declaration travels with the package, so one written before this
        -- keeps the tokenizer it was built with and opens unchanged.
        CREATE VIRTUAL TABLE passage_fts USING fts5(
            search_text, content='passage', content_rowid='id',
            tokenize="unicode61 categories 'L* N* Co Mn Mc'"
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
        -- A document that travels with the corpus and is not part of it. A
        -- corpus used as a memory sometimes has to keep an object -- a
        -- certificate, a table of coordinates -- and there was nowhere to put
        -- it: everything in the folder became passages. One such artefact of
        -- 500 vertices measured 2,003 passages, 40.6 per cent of that corpus,
        -- and made every later write cost 1.57 times as much, because packing
        -- walks the whole corpus even when one document changed.
        --
        -- It did not spoil the ranking, which was the fear and was wrong:
        -- coordinates resemble no question, so none of them reached a top five.
        -- The cost is weight and time, paid on every write thereafter.
        --
        -- Held here it is still signed, still encrypted, still one file -- and
        -- absent from the index, from the vectors, and from the passage count.
        CREATE TABLE attachment (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            pseudopath TEXT NOT NULL,
            folder TEXT,
            text TEXT NOT NULL
        );
    """)

    supplied = dates or {}

    # Read and inserted one document at a time, and not kept.
    #
    # Every document held here carries its text and its normalised form, so a
    # list of the corpus is two copies of it before the database has been
    # written at all. Measured by a consumer on a twelfth of their corpus:
    # 16.3 GB resident, which extrapolates to some 196 GB on a machine with 96.
    # They could not build the one package that would have removed their other
    # defect, which is the shape of the cost.
    #
    # What outlives a document is four things, and all four are small: the two
    # counters, the dates, and enough text from the first few to tell what
    # language the corpus is in. Attachments and documents are numbered by
    # separate counters, since they are separate tables -- setting them aside
    # first was only ever a way of keeping the document ids contiguous.
    n_documents = 0
    n_attachments = 0
    n_passages = 0
    dated: list[str] = []
    sample_parts: list[str] = []
    _t_passages = clock()

    for d in docs:
        if _is_attachment(d):
            n_attachments += 1
            connection.execute(
                "INSERT INTO attachment VALUES (?,?,?,?,?)",
                (n_attachments, d["name"], d["pseudopath"], d["folder"],
                 d["text"]))
            continue
        n_documents += 1
        i = n_documents
        d["dated"], d["dated_from"] = _date_of(d, supplied, use_mtime)
        if d.get("dated"):
            dated.append(d["dated"])
        if len(sample_parts) < 40:
            sample_parts.append(d["text"][:4000])
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
             d.get("dated"), d.get("dated_from"),
             d["norm"] if quotes == "document" else None))
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
                # Columns named rather than positional: adding one to the
                # table should not silently break the write.
                "INSERT INTO passage (id, document_id, position, text, "
                "search_text) VALUES (?,?,?,?,?)",
                # Left empty here and filled by `_index_passages`, which is
                # the only thing that reads it -- writing it now would mean
                # normalising the whole corpus twice and rewriting the table an
                # extra time for a value nothing reads in between.
                (n_passages, i, j, block, None))

    _index_passages(connection)

    from . import search as _B
    from collections import Counter as _Counter

    spent["passages"] = round(clock() - _t_passages, 3)

    df_count: _Counter = _Counter()
    # The sum and the count, not one entry per passage: the mean is all that is
    # wanted, and a list of twenty-five million integers is a list of
    # twenty-five million integers. The token counts are written in batches for
    # the same reason -- they were accumulated whole before a single row was
    # updated.
    total_length = 0
    counted_passages = 0
    with phase("terms"):
        # Read a page at a time and written between pages, never while a cursor
        # is open on the table being written. Updating a table during a scan of
        # it leaves SQLite free to show the scan rows it has already returned,
        # and the loop stops being one pass over the corpus -- which is what it
        # did: eight seconds became a quarter of an hour with no error.
        last = 0
        while True:
            page = connection.execute(
                "SELECT id, text FROM passage WHERE id > ? ORDER BY id "
                "LIMIT ?", (last, _TOKENS_BATCH)).fetchall()
            if not page:
                break
            batch: list[tuple[int, int]] = []
            for identifier, text in page:
                tk = _B.tokenize_text(_B._normalize(text))
                total_length += len(tk)
                counted_passages += 1
                batch.append((len(tk), identifier))
                for t in set(tk):
                    if indexable_term(t):
                        df_count[t] += 1
            connection.executemany(
                "UPDATE passage SET tokens = ? WHERE id = ?", batch)
            last = page[-1][0]
    connection.executemany("INSERT INTO df VALUES (?,?)", df_count.items())

    # Built from `df` and not from the passages, so the sieve and the search
    # engine share one vocabulary. It costs a tenth of a second against the tens
    # the rest of packing takes, and it is what lets a question reach a word
    # optical recognition misread.
    #
    # Asked for rather than assumed. It makes the package bigger -- a measured
    # 2.96 per cent on a corpus of books, and 17.8 on one whose vocabulary is
    # nearly all distinct -- and what it buys is recovery from transcription
    # errors, which a corpus that never went through optical recognition does
    # not have. Charging every corpus for what serves some of them is the thing
    # this project keeps declining to do.
    with phase("edges"):
        edges = _build_edges(connection) if quotes == "boundary" else 0

    with phase("expressions"):
        from . import expressions as _expressions

        summary_expressions = (_expressions.build(connection)
                               if expressions else {})

    with phase("shapes"):
        summary_shape = shapekey.build(connection) if shapes else {}
    avg_length = (total_length / counted_passages) if counted_passages else 60.0

    # The language of the corpus is recorded so that a client, a model, or the
    # query itself can tell when a question is written in another one. Retrieval
    # is lexical: a term absent from the index cannot match, and without this the
    # result is an empty answer indistinguishable from "the corpus lacks it".
    sample = " ".join(sample_parts)
    language, confidence = B.detect_language(sample)

    summary = {
        # How this package answers a quotation that runs past a passage, and
        # what that cost. A reader comparing two packages needs to know which
        # of the two shapes they have.
        "quotes": quotes,
        **({"passage_edges": edges} if edges else {}),
        **summary_expressions,
        "documents": n_documents,
        "language": language,
        "language_confidence": round(confidence, 3),
        "passages": n_passages,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_folder": folder.name,
        "mean_passage_length": round(avg_length, 2),
        "indexed_terms": len(df_count),
        **summary_shape,
    }
    if n_attachments:
        summary["attachments"] = n_attachments
    # Which document contributed most, and what share of the corpus that is. A
    # document holding 40 per cent of the passages is something whoever packed
    # it would want to see without going looking for it, and until now it took
    # a SQL query against a package they had just written.
    if n_passages:
        largest = connection.execute(
            "SELECT d.name, count(*) c FROM passage p "
            "JOIN document d ON d.id = p.document_id "
            "GROUP BY d.id ORDER BY c DESC LIMIT 1").fetchone()
        if largest:
            summary["largest_document"] = {
                "name": largest[0], "passages": largest[1],
                "share": round(largest[1] / n_passages, 4)}
    fechados = sorted(dated)
    if fechados:
        summary["dated_range"] = [fechados[0], fechados[-1]]
    # Reported whether or not any were found: "0 of 8 dated" is the signal
    # that the dates were lost on the way in, which is the defect this exists
    # to make visible.
    summary["dated_documents"] = [len(fechados), n_documents]
    manifest = folder / "_manifest.json"
    if manifest.exists():
        try:
            m = json.loads(manifest.read_text(encoding="utf-8"))
            summary["conversion"] = m.get("summary", {})
        except Exception:  # noqa: BLE001
            pass
    if semantic:
        with phase("encode"):
            summary.update(_embed_passages(connection, reuse, focus))

    for k, v in summary.items():
        connection.execute("INSERT INTO meta VALUES (?,?)",
                    (k, json.dumps(v) if not isinstance(v, str) else v))
    connection.commit()

    # Nothing to copy out: the database is the file. It used to be built in
    # memory and copied here page by page -- one copy fewer than serialising
    # it, and one more than necessary. Left as it was, that copy became a
    # backup of the file onto itself, which does not fail: it waits.
    summary["index_seconds"] = spent
    connection.commit()
    connection.close()
    return into.stat().st_size, summary

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


# How hard the package is compressed. A package is written once and read many
# times, so the default buys size with time, and the arithmetic is not linear:
# measured on a 30 MiB database of 190 documents, preset 6 takes 6.22 s for
# 1,569 KiB while preset 3 takes 1.03 s for 2,170 KiB. Six times the speed for
# 38 per cent more bytes.
#
# That trade is wrong for a package that is rewritten rather than distributed --
# a corpus used as a memory, written every time something is learned, where
# compressing and encrypting are a fixed price paid on every write because they
# are properties of the whole file. `fast` is for that case and nothing else.
#
# Preset 1 was measured too and is not the choice: 0.84 s for 3,126 KiB, barely
# faster than 3 and half again as large. Preset 0 is slower than 1 and larger,
# so it is dominated outright.
PRESET = 6
FAST_PRESET = 3


def pack(folder: Path, target: Path, key: str, issuer: str = "",
         signing_key: str = "", semantic: bool = False,
         reuse_from: Path | None = None,
         focus: list[str] | None = None,
         dates: dict | None = None, use_mtime: bool = False,
         fast: bool = False, shapes: bool = False,
         preset: int | None = None,
         compression: str = DEFAULT_COMPRESSION,
         quotes: str = "document",
         expressions: bool = False) -> dict:
    """Write the .mdcx file and return its figures.

    `preset` is the LZMA level, 0 to 9, and overrides `fast` when given. The
    two constants `fast` chooses between are a decision about a distributed
    package -- compressed once and downloaded many times -- and there is a
    second case they cannot express: a corpus rebuilt whenever it grows, where
    the clock matters and the bytes do not. Measured by a consumer over 120 MiB
    of their own text, preset 0 wrote in 5.9 s against 24.5 s at preset 3, for
    30.0% of the original against 26.1%. Which of those is right is the
    caller's to decide, not this module's.

    The preset travels inside the compressed stream, so a package written at
    any level is opened by any reader.

    `quotes` decides how a quotation that runs past the end of a passage is
    found: "document" keeps the whole normalised text of every document, which
    has no length limit and costs a second copy of the corpus; "boundary"
    indexes the join between consecutive passages instead, which finds a
    quotation spanning one cut and not one longer than the join. Measured on
    one package, that copy was 257.4 MB of a 783.8 MB database.

    `compression` names what compresses the body: "lzma" or "zstd". The header
    records it, and a package is opened by what its header names, so writing
    one with zstd does not change how anything else is read. Reading a zstd
    package needs the `zstandard` module; LZMA needs nothing.
    """
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
    inherited_focus = False
    if semantic and reuse_from is not None:
        from . import semantic as _S

        # Reading the previous package needs the same key, which the caller
        # already holds: a package that cannot be decrypted holds no vectors
        # that can be reused. Both halves come out of the one opening.
        reuse, previous = _reusable(Path(reuse_from), key, _S.model_name())

        # A calibration outlives the vectors it was measured with. Reusing a
        # package recovered the expensive half and dropped the cheap one, and
        # the drop was invisible: the threshold stored barely moves -- 0.6393
        # to 0.6316 in the case that found this -- while the margin applied to
        # it goes from 0.95 to 0.60, so the effective threshold falls by a
        # third and a corpus that grows loses its calibration on its first
        # update. Passing --focus again still overrides this, so recalibrating
        # against different questions works exactly as before.
        if focus is None and previous:
            focus = previous
            inherited_focus = True

    # Two scratch files beside the target rather than in the system's temporary
    # folder: a package of a large corpus is large, and the disk with room for
    # it is the one the caller chose.
    workspace = tempfile.mkdtemp(prefix=".mdcx-pack-", dir=str(target.parent))
    built = Path(workspace) / "database"
    sealed = Path(workspace) / "body"

    t0 = time.perf_counter()
    bytes_database, summary = _build_database(folder, built, semantic=semantic,
                                              reuse=reuse, focus=focus,
                                              dates=dates, use_mtime=use_mtime,
                                              shapes=shapes, quotes=quotes,
                                              expressions=expressions)
    t_base = time.perf_counter() - t0

    # An empty package is written without complaint and fails only when queried,
    # long after the mistake. The usual cause is pointing at the folder of source
    # documents rather than at the converted Markdown.
    if not summary["documents"]:
        shutil.rmtree(workspace, ignore_errors=True)
        raise ValueError(
            f"No Markdown documents found in {folder}. "
            "This should be the output folder of a conversion, not the source documents."
        )

    salt = os.urandom(16)
    derived_key = _derive_key(key, salt)

    # Compressed, encrypted and hashed in one pass over the database, block by
    # block. The two stages used to be separate and each held a whole copy of
    # the corpus; timing them apart is no longer possible and no longer worth
    # the memory it cost.
    t0 = time.perf_counter()
    if preset is None:
        level = FAST_PRESET if fast else PRESET
    elif not 0 <= preset <= 9:
        # Said rather than clamped: a caller who asked for 12 has a reason to
        # think it exists, and quietly writing at 9 would leave them measuring
        # a level they did not choose.
        raise ValueError(f"preset must be between 0 and 9, not {preset}")
    else:
        level = preset
    if quotes not in QUOTE_STRATEGIES:
        raise ValueError(
            f"quotes must be one of {', '.join(QUOTE_STRATEGIES)}, "
            f"not {quotes!r}")
    if compression not in COMPRESSORS:
        raise ValueError(
            f"compression must be one of {', '.join(COMPRESSORS)}, "
            f"not {compression!r}")
    if compression == "zstd":
        _zstd()   # refused here rather than after the corpus was indexed
    nonce, body_digest, bytes_body = _seal(built, sealed, derived_key, level,
                                           compression)
    t_seal = time.perf_counter() - t0
    built.unlink(missing_ok=True)

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
        "compression": compression,
        # In the header because it decides what a reader can ask of the
        # package, and the header is read without the key.
        "quotes": summary.get("quotes", "document"),
        "salt": salt.hex(),
        "nonce": nonce.hex(),
        "body_digest": body_digest,
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
        # In the header because a package whose weight is mostly an attachment
        # is a different proposition from one of the same size that is all
        # corpus, and that is decided before opening it.
        **({"attachments": summary["attachments"]}
           if summary.get("attachments") else {}),
    }
    if signing_key:
        from cryptography.hazmat.primitives.asymmetric import ed25519

        private = ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(signing_key))
        header["signature"] = _sign(header["body_digest"], signing_key)
        header["public_key"] = private.public_key().public_bytes_raw().hex()

    encoded_header = json.dumps(header, ensure_ascii=False).encode("utf-8")

    # The header carries the digest of the body, so the body has to exist
    # before the header can be written. It is copied in rather than held: the
    # point of the whole change is that no stage holds the corpus.
    with open(target, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(encoded_header)))
        f.write(encoded_header)
        with open(sealed, "rb") as body_file:
            shutil.copyfileobj(body_file, f, SEAL_BLOCK_BYTES)
    shutil.rmtree(workspace, ignore_errors=True)

    return {
        "bytes_database": bytes_database,
        "bytes_compressed": bytes_body,
        # Which level actually wrote it, since three things can
        # decide it and a reader comparing two packages needs to
        # know they were written the same way.
        "compression_preset": level,
        "compression": compression,
        "bytes_file": target.stat().st_size,
        "seconds_index": round(t_base, 2),
        # What that number is made of. One figure for indexing hid which part
        # of it a caller could act on: reading the folder, cutting passages,
        # counting terms, building the shape index, encoding. A consumer
        # measuring 355 s of indexing attributed it to the model, which was a
        # small part of it.
        "seconds_index_by_phase": summary.get("index_seconds", {}),
        # Compressing and encrypting are properties of the whole file, so they
        # cost the same whether one document was added or the corpus was
        # rebuilt. That is a fixed price per write, and it grows with the
        # corpus rather than with what was added -- reported here so a caller
        # writing often can decide how often to write, and use --fast when the
        # package is being rewritten rather than distributed.
        "seconds_seal": round(t_seal, 2),
        # Said out loud, because inheriting it silently would be the same fault
        # as dropping it silently, only in the other direction.
        **({"focus_inherited": True} if inherited_focus else {}),
        **summary,
    }


def _is_attachment(document: dict) -> bool:
    """Whether the document asked not to be indexed.

    Declared by the document itself rather than by a folder, because what
    decides is what the thing is, and that travels with it: a certificate moved
    between folders is still a certificate. `indexed: false` in the front
    matter, and nothing else -- an absent field means an ordinary document,
    which is what every package written before this has.
    """
    for line in (document.get("front_matter") or "").splitlines():
        lowered = line.strip().lower()
        if lowered.startswith("indexed:"):
            return lowered.split(":", 1)[1].strip().strip('"') in (
                "false", "no", "off", "0")
    return False


def attachments(connection: sqlite3.Connection) -> list[dict]:
    """The documents kept with the corpus and left out of it.

    Empty for a package written before attachments existed, which is the
    answer rather than an error.
    """
    try:
        rows = connection.execute(
            "SELECT name, pseudopath, folder, text FROM attachment "
            "ORDER BY id").fetchall()
    except sqlite3.OperationalError:
        return []
    return [{"name": n, "pseudopath": p, "folder": f, "text": t}
            for n, p, f, t in rows]


def calibrate(path: Path, key: str, questions: list[str],
              signing_key: str = "") -> dict:
    """Measure a closed package against questions, and leave the measure in it.

    The threshold was writable only by `pack`, and `pack` walks a folder of
    documents. So a package whose source material is gone could never be
    calibrated: it falls back to a constant nobody measured on it, and stays
    there in every future version. Measured on four such packages, that constant
    cuts through the middle of the range the corpus answers -- 38 of 48 domain
    questions accepted against 47 of 48 with a threshold taken from questions.

    Nothing about calibrating needs the material. It needs the vectors, which
    are already inside, and questions, which come from outside; the function
    that measures it touches nothing else. Only writing the result did.

    The package is rewritten in place: same documents, same passages, same
    vectors, re-compressed and re-encrypted around a changed `meta`. It costs
    what a write costs and nothing more.

    A signed package needs its signing key to stay signed. Rather than quietly
    returning an unsigned package where a signed one went in -- the exact shape
    of fault these reports keep finding -- it refuses and says so.
    """
    import lzma

    if not questions:
        raise ValueError("calibrating needs the questions to calibrate against")

    header = read_header(Path(path))
    if header.get("signature") and not signing_key:
        raise ValueError(
            "this package is signed, and rewriting it would break the "
            "signature. Pass the signing key to sign the result, or verify "
            "and re-issue it deliberately")

    connection, _ = open_package(Path(path), key)
    try:
        if not has_vectors(connection):
            raise ValueError(
                "calibrating measures how near the corpus comes to a question, "
                "which is a cosine: this package carries no meaning index. "
                "Pack it with --multilingual")

        reach = _answerable_at_focus(connection, questions)
        if reach is None:
            raise ValueError(
                "none of the questions could be measured against this package")

        for k, v in (("answerable_at", json.dumps(reach)),
                     ("answerable_at_from", "focus-after"),
                     ("focus", json.dumps(list(questions)))):
            connection.execute("DELETE FROM meta WHERE key = ?", (k,))
            connection.execute("INSERT INTO meta VALUES (?,?)", (k, v))
        connection.commit()
        data = bytes(connection.serialize())
    finally:
        connection.close()

    compressed = lzma.compress(data, preset=PRESET)
    salt = os.urandom(16)
    nonce, body = _encrypt(compressed, _derive_key(key, salt))

    header = dict(header)
    header.pop("_intact", None)
    header.update({"salt": salt.hex(), "nonce": nonce.hex(),
                   "body_digest": hashlib.sha256(body).hexdigest(),
                   "answerable_at": reach,
                   "answerable_at_from": "focus-after",
                   "signature": "", "public_key": ""})
    if signing_key:
        from cryptography.hazmat.primitives.asymmetric import ed25519

        private = ed25519.Ed25519PrivateKey.from_private_bytes(
            bytes.fromhex(signing_key))
        header["signature"] = _sign(header["body_digest"], signing_key)
        header["public_key"] = private.public_key().public_bytes_raw().hex()

    encoded_header = json.dumps(header, ensure_ascii=False).encode("utf-8")
    target = Path(path)
    # Written beside and moved over, so an interrupted write leaves the package
    # that was there rather than half of a new one.
    scratch = target.with_suffix(target.suffix + ".calibrating")
    with open(scratch, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(encoded_header)))
        f.write(encoded_header)
        f.write(body)
    os.replace(scratch, target)

    return {"answerable_at": reach, "answerable_at_from": "focus-after",
            "questions": len(questions), "signed": bool(signing_key)}


def add_shapes(path: Path, key: str, signing_key: str = "") -> dict:
    """Add the shape index to a package that was written without one.

    Same shape as calibrating: a package is rewritten around a changed database
    rather than rebuilt, so the documents, the passages and the vectors are the
    ones that were there. Nothing is converted again.

    A signed package needs its signing key, and refuses without it instead of
    coming back unsigned.
    """
    import lzma

    from . import shapekey as _shape

    header = read_header(Path(path))
    if header.get("signature") and not signing_key:
        raise ValueError(
            "this package is signed, and rewriting it would break the "
            "signature. Pass the signing key to sign the result, or verify "
            "and re-issue it deliberately")

    connection, _ = open_package(Path(path), key)
    try:
        added = _shape.build(connection)
        data = bytes(connection.serialize())
    finally:
        connection.close()

    compressed = lzma.compress(data, preset=PRESET)
    salt = os.urandom(16)
    nonce, body = _encrypt(compressed, _derive_key(key, salt))

    header = dict(header)
    header.pop("_intact", None)
    header.update({"salt": salt.hex(), "nonce": nonce.hex(),
                   "body_digest": hashlib.sha256(body).hexdigest(),
                   "signature": "", "public_key": ""})
    if signing_key:
        from cryptography.hazmat.primitives.asymmetric import ed25519

        private = ed25519.Ed25519PrivateKey.from_private_bytes(
            bytes.fromhex(signing_key))
        header["signature"] = _sign(header["body_digest"], signing_key)
        header["public_key"] = private.public_key().public_bytes_raw().hex()

    encoded_header = json.dumps(header, ensure_ascii=False).encode("utf-8")
    target = Path(path)
    scratch = target.with_suffix(target.suffix + ".shaping")
    with open(scratch, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(encoded_header)))
        f.write(encoded_header)
        f.write(body)
    os.replace(scratch, target)
    return {**added, "signed": bool(signing_key)}


def _index_passages(connection: sqlite3.Connection) -> None:
    """Build the word index, then stop storing the copy it was built from.

    `passage.search_text` is the searchable form of each passage -- folded case
    and accents, and split into characters for the scripts that do not separate
    words. FTS5 is declared over it with external content, so it reads the
    column while indexing and never again: a query runs on the index, and the
    text a reply quotes comes from `passage.text`.

    So after the index exists the column is duplication, and a measured 7.2 per
    cent of the file. Emptied, every one of eighteen queries -- short and long
    -- returns exactly what it returned before.

    What emptying it would break is a later `rebuild`, which would read nothing
    and leave a corpus that answers no question at all, without an error. That
    is why the two happen in one place: whoever needs to index again calls this,
    which fills the column, rebuilds, and empties it once more. It is derived
    from `text` and costs a pass over the passages.
    """
    from . import search as B

    rows = [(B._normalize(B.segment_for_index(text)), rowid)
            for rowid, text in connection.execute("SELECT id, text FROM passage")]
    connection.executemany(
        "UPDATE passage SET search_text = ? WHERE id = ?", rows)
    connection.execute("INSERT INTO passage_fts(passage_fts) VALUES('rebuild')")
    # Emptied rather than dropped: the column is what FTS5 was declared over,
    # and a schema that no longer matches the declaration is a different kind of
    # trouble. What is given back is the bytes.
    connection.execute("UPDATE passage SET search_text = NULL")
    # And the pages the text used to occupy are handed back. Without this the
    # column is emptied and the file does not shrink at all -- SQLite keeps the
    # freed pages on its free list with the old bytes still in them, so the
    # serialised database still carries the text and the compressor still has to
    # encode it. Measured: emptying alone made the package 1.4 per cent LARGER.
    connection.commit()
    connection.execute("VACUUM")


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
    return _reusable(path, key, model)[0]


def stored_focus(path: Path, key: str) -> list[str] | None:
    """The questions a package was calibrated against, if it was."""
    return _reusable(path, key, model=None)[1]


def _reusable(path: Path, key: str,
              model: str | None) -> tuple[dict[str, bytes], list[str] | None]:
    """What a package can hand to the one being built from it.

    Two things, read in one opening. The vectors are the expensive half and the
    reason `--reuse` exists; the questions it was calibrated against are the
    cheap half, and losing them costs more than losing the vectors -- vectors
    are recomputed at a price, a calibration that silently reverts to being
    estimated from passages is not recovered at all.
    """
    connection, _ = open_package(path, key)
    try:
        stored = connection.execute(
            "SELECT value FROM meta WHERE key = 'focus'").fetchone()
        focus = None
        if stored and stored[0]:
            try:
                focus = json.loads(stored[0]) or None
            except (ValueError, TypeError):
                focus = None

        if model is None:
            return {}, focus

        # The model is recorded in the package metadata rather than in the
        # header, which is the part readable without the key.
        row = connection.execute(
            "SELECT value FROM meta WHERE key = 'embedding_model'").fetchone()
        if not row or row[0] != model:
            # A different model: the vectors are unusable, and the questions
            # are not. They are text, and they calibrate whatever encodes them.
            return {}, focus
        rows = connection.execute(
            "SELECT p.text, v.vector FROM passage p "
            "JOIN passage_vector v ON v.passage_id = p.id").fetchall()
        return {passage_digest(text): vector for text, vector in rows}, focus
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


def calibrated_from_questions(connection: sqlite3.Connection) -> bool:
    """Whether this package's reach was measured against real questions.

    It changes what the number means, which is why it is worth asking rather
    than assuming. Estimated from passages, the reach is how near a question the
    corpus answers would come, and a share of it becomes the threshold. Taken
    from the questions themselves it already is the threshold -- the lowest any
    of them reached -- and scaling it again would put the cut well below
    anything that was measured.
    """
    try:
        row = connection.execute(
            "SELECT value FROM meta WHERE key='answerable_at_from'").fetchone()
    except Exception:  # noqa: BLE001
        return False
    if not row or not row[0]:
        return False
    # Meta stores a string as itself and everything else as JSON, so this value
    # is the bare word `focus`. Reading it back through json.loads() raised on
    # every calibrated package -- and the quoted form is accepted too, since
    # nothing but this comparison depends on which was written.
    #
    # `focus-after` is the same measurement taken on a closed package rather
    # than while packing it. The distinction is worth keeping in the record --
    # one was measured over the corpus as it was built, the other over the
    # corpus as it stands -- and it makes no difference to the margin, because
    # both are thresholds taken from questions rather than estimates from
    # passages.
    return str(row[0]).strip().strip('"').startswith("focus")


# Which passage stands in for the tail. Far enough down that a handful of real
# answers do not drag it up, near enough that it is still the same corpus.
TAIL_AT = 50

# What share of its own reach a corpus has to come, for the question to count as
# one it is about. Measured over four packages built for this: the worst
# question the corpus answers reaches 0.68 of its reach and the best unrelated
# one 0.53, so the cut goes between them.
#
# This is a constant about how questions relate to corpora, applied to a number
# each corpus measured about itself -- which is what an absolute threshold could
# never be, and why one landed inside the answered range of one corpus and below
# another's.
ANSWERS_AT_SHARE = 0.60

# The same idea for a package that was given its questions, where the reach is
# already the threshold rather than an estimate of one: the lowest any of the
# declared questions reached.
#
# Not 1.0, which is where the arithmetic points and where it fails. Setting the
# cut exactly at the worst declared question marks that very question, because
# the reach is stored rounded to four places and the cosine computed at query
# time falls a little either side of it. This is the width of that rounding and
# of the fine variation around it, and no more.
ASKED_MARGIN = 0.95

# The pair that decides for a package that measured no reach of its own, which
# is every package built before that was measured. Both are required: on a small
# corpus the first fires wrongly and the second holds it back, on a large one the
# second fires wrongly and the first holds it back. The long account of why
# neither was moved is with the warning that uses them, in mcp_server.
NOTHING_NEAR = 0.635
STANDS_CLEAR = 0.25


def _below_reach(closeness: float, clearance: float, reach: float | None,
                 from_questions: bool = False) -> bool:
    """Whether a corpus comes too near nothing to count as being about this.

    One rule, in one place. It used to live only in the MCP server, where the
    warning is raised, so anything else that wanted to branch on it -- a
    consumer serving several packages with different roles, deciding which one
    answers -- had to rebuild it out of private functions and a constant
    imported from a server it was not otherwise using.

    Choosing the margin is the part that is easiest to get wrong, and getting it
    wrong raises no error: applying the passage share to a threshold calibrated
    from questions was measured letting seven of eight unrelated queries
    through.
    """
    if reach:
        return closeness < reach * (ASKED_MARGIN if from_questions
                                    else ANSWERS_AT_SHARE)
    return closeness < NOTHING_NEAR and clearance < STANDS_CLEAR


def closeness(connection: sqlite3.Connection,
              text: str) -> tuple[float, float] | None:
    """How near this package comes to a text, and how far that stands clear.

    The best cosine against the package, and its distance from the tail of the
    ranking. None where there is no such number -- a package that indexes words
    alone has none, and inventing one from BM25 would be the mistake this
    replaced: a BM25 score depends on the corpus it was measured in, so two
    packages cannot be compared by it. A cosine means the same thing in every
    package, because the model that computes it knows nothing of the corpus the
    passage was drawn from.

    Per package rather than over all of them, which is the whole point of it
    being here. Whether *anything* open is about a question is a property of the
    set and is what the MCP warning asks; which package answers is not, and no
    aggregate can be taken apart afterwards.
    """
    scores = cosines(connection, text)
    if not scores:
        return None
    tail = scores[min(TAIL_AT, len(scores) - 1)]
    return _a_cosine(scores[0]), float(scores[0] - tail)


def _a_cosine(value: float) -> float:
    """Hold a cosine inside the range a cosine has.

    Vectors are stored in half precision, and rounding to it costs the
    normalisation: a vector written with norm 1 comes back with norm 1 +/- 2e-4,
    so a dot product against a query computed in single precision leaves the
    interval. Four separately built packages all topped out at exactly 1.000428,
    which is what says it is the format and not the text.

    Nothing decided on this changes -- thresholds sit between 0.55 and 0.59 and
    the differences that separate groups are hundredths, three orders of
    magnitude above the excess. What breaks is the invariant, and an invariant
    that almost holds is worse than one that does not: a consumer asserting
    `0 <= c <= 1` fails one time in many, in production, and a project that saw
    the symptom found a plausible cause for it and stopped looking -- the digit
    above 1 went unexplained across two rounds of measurement.
    """
    return float(min(1.0, max(-1.0, value)))


def cosines(connection: sqlite3.Connection, text: str) -> list[float]:
    """Every passage of this package scored against the text, best first.

    The raw material of both questions that get asked of it: which package
    answers, decided per package, and whether anything open is about the
    question at all, which pools the scores of several packages before looking
    at the tail. Empty where the package has no meaning index.
    """
    if not _semantic_ready(connection):
        return []
    try:
        import numpy as np

        from . import semantic as S

        vector = np.asarray(S.encode([text], role="query")[0], dtype=np.float32)
    except Exception:  # noqa: BLE001
        return []
    return cosines_of(connection, vector)


def cosines_of(connection: sqlite3.Connection, vector) -> list[float]:
    """The same, for a query already encoded.

    Separate because encoding is the expensive half and does not depend on the
    package: a caller asking several packages about one question encodes it
    once. Asking `cosines` per package instead would pay for that encoding
    again for each of them.
    """
    try:
        import numpy as np

        identifiers, matrix = _vectors(connection)
        if not identifiers:
            return []
        # Both operands, not one. The stored side was repaired first and the
        # excess stayed, because it came from the query: a vector arriving here
        # is whatever the caller had, and a cosine needs unit length on both.
        norm = float(np.linalg.norm(vector))
        if norm and abs(norm - 1.0) > 1e-6:
            vector = np.asarray(vector, dtype=np.float32) / norm
        return sorted((_a_cosine(v) for v in (matrix @ vector).tolist()),
                      reverse=True)
    except Exception:  # noqa: BLE001
        return []


def answers(connection: sqlite3.Connection, text: str) -> bool:
    """Whether this package is about the text, by its own calibration.

    False for a package with no meaning index, which is honest rather than
    conservative: without vectors there is no number, and a decision that cannot
    be made should not be made optimistically.
    """
    measured = closeness(connection, text)
    if measured is None:
        return False
    near, clear = measured
    return not _below_reach(near, clear, answerable_at(connection),
                            calibrated_from_questions(connection))


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

    # The vectors were unit length when they were written and are not when they
    # are read back: rounding to half precision costs the normalisation, leaving
    # norms of 1 +/- 2e-4. Restored here, once per package rather than once per
    # query, because the cache holds the result -- so it costs nothing per query
    # and everything computed from the matrix is a cosine again.
    #
    # It changes no ranking: every row is scaled by about the same amount. What
    # it fixes is that the numbers handed out stopped satisfying the definition
    # of the thing they are called, which was reported after a cosine of
    # 1.000428 sent a reader looking for an explanation in the wrong place.
    if len(rows):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        np.divide(matrix, norms, out=matrix, where=norms > 0)

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
    connection.deserialize(_decompress(compressed, header.get("compression")))
    connection.digest = header["body_digest"]
    return connection, header

_SQL_TEMPLATE = """
    SELECT d.name, d.pseudopath, d.source, p.text, bm25(passage_fts) AS score,
           p.id AS passage_id, {length}{dates}
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


def _documents_holding(connection: sqlite3.Connection, needle: str,
                       names: list[str]) -> list[str]:
    """Which of these documents contain this phrase, read from their own text.

    Asked of the documents already in the ranking, and of no others. This
    branch reorders them; a document it found outside the ranking could not
    appear in the reply, so reading the whole corpus to find one was work with
    nowhere to go -- and `normalized_text` is the corpus again.
    """
    found: list[str] = []
    with _CONNECTION_LOCK:
        for start in range(0, len(names), _TERMS_PER_STATEMENT):
            batch = names[start:start + _TERMS_PER_STATEMENT]
            placeholders = ",".join("?" * len(batch))
            found.extend(
                name for (name, text) in connection.execute(
                    "SELECT name, normalized_text FROM document "
                    f"WHERE name IN ({placeholders})", batch)
                if text and needle in text)
    return found


def _documents_joining(connection: sqlite3.Connection, needle: str) -> list[str]:
    """Which documents hold this phrase across a cut, by the indexed join.

    The rowid of a match is the passage that opens the join, which is a passage
    of the corpus; the joined text is not, and is never given back as one.

    A phrase this way is found by index rather than by reading, so unlike the
    branch above it does not need to be told which documents to look at.
    """
    try:
        with _CONNECTION_LOCK:
            rows = connection.execute(
                "SELECT d.name FROM passage_edge_fts e "
                "JOIN passage p ON p.id = e.rowid "
                f"JOIN document d ON d.id = p.{document_column(connection)} "
                "WHERE passage_edge_fts MATCH ?",
                ['"' + needle.replace('"', "") + '"']).fetchall()
    except sqlite3.OperationalError:
        return []
    return [name for (name,) in rows]


def _by_expression(connection: sqlite3.Connection, stated: dict,
                   limit: int, only: str | None) -> list[dict]:
    """The passages that state these expressions, as a reply.

    Ranked by how many of the question's expressions a passage carries, which
    is the whole of what can be said about a match that has no words to weigh.
    """
    per_passage: dict[int, list[str]] = {}
    for expression, passages in stated.items():
        for identifier in passages:
            per_passage.setdefault(identifier, []).append(expression)
    if not per_passage:
        return []

    ids = sorted(per_passage, key=lambda i: -len(per_passage[i]))[:limit]
    marks = ",".join("?" * len(ids))
    sql = ("SELECT p.id, d.name, d.pseudopath, d.source, p.text "
           "FROM passage p JOIN document d ON d.id = p."
           + document_column(connection)
           + f" WHERE p.id IN ({marks})")
    params: list = list(ids)
    if only:
        sql += " AND d.source = ?"
        params.append(only.upper())
    with _CONNECTION_LOCK:
        rows = connection.execute(sql, params).fetchall()

    order = {identifier: n for n, identifier in enumerate(ids)}
    rows.sort(key=lambda r: order.get(r[0], len(order)))
    return [{"document": r[1], "pseudopath": r[2], "source": r[3],
             "passage": r[4],
             # Named apart from a lexical score, which it is not: nothing was
             # weighted by rarity because there were no words to weigh.
             "score": float(len(per_passage[r[0]])),
             "terms": sorted(per_passage[r[0]]),
             "matched_by": "expression"} for r in rows]


def _run_match(connection: sqlite3.Connection, expr: str, limit: int,
              only: str | None) -> list[dict]:
    fechas = ", d.dated, d.dated_from" if has_dates(connection) else ""
    # The stored token count where the package has one, so the scoring loop
    # does not have to read the passage back to find out how long it is.
    length = "p.tokens" if _has_token_counts(connection) else "NULL"
    sql = _SQL_TEMPLATE.format(document_column=document_column(connection),
                               dates=fechas, length=length)
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
                "passage": r[3], "score": round(-r[4], 3),
                # Under private names: they are how the scoring loop reaches
                # the index, and they are removed before the reply is built.
                "_passage_id": r[5], "_tokens": r[6]}
        if fechas:
            item["dated"], item["dated_from"] = r[7], r[8]
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

def terms_of(query_text: str) -> list[str]:
    """The terms a query will be weighed by, without running it.

    Published because a caller serving several packages has to gather the
    statistics for exactly these terms before any package is asked, and
    reproducing the rule outside would be a second definition of it.
    """
    from . import search as B

    return B.searchable_terms(B._normalize(query_text))


def corpus_statistics_over(connections, terms) -> tuple[int, float, dict]:
    """The corpus figures of several packages taken together.

    What a caller serving more than one package needs, and could not compute:
    BM25 weighs a term by how rare it is, and rarity is a property of the
    corpus the score was computed over. Two packages therefore score the same
    passage differently, and the two numbers have no common meaning.

    Serving several packages is exactly where that matters, because the answer
    is assembled from all of them. Returns the passage count, the mean passage
    length weighted by how many passages each package holds -- an unweighted
    average would let a package of eleven documents count as much as one of
    five thousand -- and the document frequency of each term summed across
    them.
    """
    total = 0
    weighted = 0.0
    frequencies: dict[str, int] = {}
    for connection in connections:
        passages, mean = _corpus_scale(connection)
        total += passages
        weighted += mean * passages
        for term, count in _frequencies_for(connection, terms).items():
            frequencies[term] = frequencies.get(term, 0) + count
    return (max(total, 1),
            (weighted / total) if total else 60.0,
            frequencies)


def lexical_query(connection: sqlite3.Connection, query_text: str, limit: int = 8,
              only: str | None = None, notes: dict | None = None,
              corpus: tuple | None = None) -> list[dict]:
    """Resolve a query by word, ranking by document rather than by isolated passage.

    `corpus` is `(passages, mean_length, frequencies)` to weigh the terms
    against, instead of this package's own. BM25 weighs a term by how rare it is
    in the corpus it is computed over, so two packages give the same passage
    different scores and the numbers cannot be ordered together. A caller
    serving several packages passes the same figures to all of them, and the
    scores become comparable -- see `corpus_statistics_over`.
    """
    from . import search as B

    phrase = query_text.strip().split(".")[0][:160].strip()
    effective = phrase if len(phrase.split()) >= 5 else query_text

    asked = B.searchable_terms(B._normalize(effective))
    terms = B.expand_terms(list(asked), _corpus_language(connection))

    # The expressions of the question, where the package indexed them. Word
    # matching discards the symbols, so `pq | b(b+p+q)` and `pq | b(b-p-q)`
    # reduce to the same terms -- and asked on their own, to none at all. A
    # corpus of mathematics asked about a statement is asking about exactly
    # what the tokenizer removed.
    from . import expressions as _expressions

    stated: dict = {}
    if _expressions.has_index(connection):
        wanted = _expressions.extract(effective)
        if wanted:
            with _CONNECTION_LOCK:
                stated = _expressions.passages_stating(connection, wanted)

    if not terms:
        # Nothing to match on words. If the question carried an expression the
        # corpus states, that is the answer; before this it was silence.
        return (_by_expression(connection, stated, limit, only)
                if stated else [])
    distinct_terms = set(terms)

    expr = " OR ".join(f'"{B.segment_for_index(t)}"' for t in distinct_terms)
    candidates = _run_match(connection, expr, CANDIDATES, only)
    widened: dict = {}
    if not candidates:
        # Only here, where the answer is already empty and there is nothing left
        # to lose. A word optical recognition misread is a word the index does
        # not hold, so no amount of matching reaches it; the shape of its
        # letters proposes what it might have been, and edit distance decides.
        #
        # The scoring terms are widened along with the expression, and that is
        # not incidental: the passage found this way contains the misread word
        # and not the word that was asked for, so a frequency count over the
        # original terms alone comes back empty and every result is dropped one
        # loop later. Widening the query without widening what is counted looks
        # like the sieve finding nothing.
        from . import shapekey as _shape

        # Over what was asked, not over what `expand_terms` added. The
        # glossary appends cross-language equivalents, which a monolingual
        # corpus does not hold either -- widening those would report, as a word
        # of the question the corpus lacks, a word nobody typed.
        widened = _shape.expand(connection, asked)
        if not widened:
            return []
        distinct_terms = distinct_terms | {
            c for found in widened.values() for c in found}
        expr = " OR ".join(f'"{B.segment_for_index(t)}"' for t in distinct_terms)
        candidates = _run_match(connection, expr, CANDIDATES, only)
        if not candidates:
            return []
        if notes is not None:
            # Recorded here and not when the widening was computed. Proposing a
            # candidate is not finding a passage: the second match can still
            # come back empty -- `direction` scopes it while the vocabulary it
            # was drawn from does not -- and a note surviving that would tell a
            # reader that passages the semantic engine found were read under
            # another spelling, which they were not.
            notes["read_as"] = {t: list(c) for t, c in widened.items()}

    if corpus is not None:
        n_passages, avg_length, df = corpus
    else:
        n_passages, avg_length = _corpus_scale(connection)
        df = _frequencies_for(connection, distinct_terms)

    # What each candidate holds of the question, from the index that already
    # recorded it. Half of a lexical query was spent reading the candidate
    # passages back and tokenising them for counts FTS5 had taken when the
    # package was built. Where the index cannot answer -- an older package, an
    # unusual one -- this comes back None and each passage is read as before.
    seen = _occurrences(connection, distinct_terms,
                        [r["_passage_id"] for r in candidates])

    by_document: dict[str, list[dict]] = {}
    for r in candidates:
        if seen is not None and r.get("_tokens"):
            freq = seen.get(r["_passage_id"], {})
            length = max(int(r["_tokens"]), 1)
        else:
            # Tokenised once. The frequencies and the length are two readings
            # of the same list, and taking them apart normalised every
            # candidate passage twice for one answer.
            tokens = B.tokenize_text(B._normalize(r["passage"]))
            freq = {}
            for t in tokens:
                if t in distinct_terms:
                    freq[t] = freq.get(t, 0) + 1
            length = max(len(tokens), 1)
        if not freq:
            continue
        score = 0.0
        for t, f in freq.items():
            d_t = df.get(t, 1)
            idf = math.log(1 + (n_passages - d_t + 0.5) / (d_t + 0.5))
            score += idf * (f * (K1 + 1)) / (
                f + K1 * (1 - B_LENGTH + B_LENGTH * length / avg_length))
        r = dict(r)
        r["score"] = round(score, 3)
        r["terms"] = sorted(freq)
        # The two private keys were how this loop reached the index; they are
        # not part of what a caller is given.
        r.pop("_passage_id", None)
        r.pop("_tokens", None)
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
        # Asked of the documents already in the ranking, and of no others. What
        # this branch does is reorder them; a document it found outside the
        # ranking could not appear in the reply, so reading the whole corpus to
        # find one was work with nowhere to go. The column exists to catch a
        # quotation that straddles the cut between two passages, which the
        # passage index cannot see -- that still works, over the candidates.
        #
        # It is not a small saving at scale: `normalized_text` is the corpus
        # again, 257 MB in one package of a collection whose whole is 17 GB.
        preferred = (_documents_joining(connection, needle)
                     if _has_edges(connection)
                     else _documents_holding(connection, needle,
                                             [passages[0]["document"]
                                              for _score, passages in ranking]))
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
          prefer: str | None = None, notes: dict | None = None,
          corpus: tuple | None = None) -> list[dict]:
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
        connection, query_text, limit * 3, only, notes=notes, corpus=corpus)
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

# The shortest term the index records. Two characters carry too little to
# discriminate in the scripts that separate words, and the table would fill with
# them; a single character carries a word in the ones that do not, which is the
# exception below it.
#
# It is worth naming rather than leaving inline because it is what makes a term
# absent from `df` ambiguous: absent can mean the corpus never saw it, or it can
# mean the indexer was never going to record it. Anything weighing terms by
# rarity has to tell those apart -- an absent term takes the maximum idf, so
# reading absence as novelty makes the shortest, emptiest words the most
# informative ones.
MINIMUM_TERM_LENGTH = 3


def indexable_term(term: str) -> bool:
    """Whether this term is one the index would record at all."""
    from . import search as B

    return len(term) >= MINIMUM_TERM_LENGTH or (len(term) == 1 and B._is_cjk(term))


def vocabulary(connection: sqlite3.Connection) -> dict:
    """What this package knows about words, and by what rule.

    `df` alone is not enough to ask whether a text brings new vocabulary,
    because absence from it has two meanings. This carries the rule that
    produced it alongside the counts, so the two can be told apart.

    The keys of `df` are normalised -- folded case, folded accents -- so a
    caller tokenising with `search.tokenize_text` gets `GPU` where the table
    holds `gpu`, and has to normalise before looking a term up. `unknown_terms`
    does that; anyone reading `df` directly must do it themselves.
    """
    df, passages, mean_length = corpus_statistics(connection)
    return {
        "df": df,
        "terms": len(df),
        "passages": passages,
        "mean_passage_length": mean_length,
        "minimum_term_length": MINIMUM_TERM_LENGTH,
        "normalized": True,
    }


def unknown_terms(connection: sqlite3.Connection, text: str) -> list[str]:
    """The terms of a text this corpus has genuinely never seen.

    What is excluded is the difference that matters: a term the indexer would
    not have recorded whatever the corpus contained is not new, it is invisible,
    and counting it as new was measured turning a novelty score into a detector
    of short function words -- seven of eight questions a corpus answers well
    declared between 0.37 and 0.55 of unknown vocabulary, and fall to exactly
    zero once the rule is applied.

    Order is kept and repeats are dropped, so the result reads as the unfamiliar
    part of the text in the order it was written.

    It also cannot see a short token, and that has bitten a reader. The index
    records nothing under three characters, so a number like `10` or `6` is not
    absent from the vocabulary -- it was never in it, and never can be. A
    consumer measuring novelty by unfamiliar words therefore cannot tell two
    instances of one problem apart when what distinguishes them is a small
    number, and no weighting fixes it: the token is not there to weigh.

    This is literal, and the index it reads is not the one that crosses
    languages. The meaning index reaches a Spanish question against an English
    corpus; the word index cannot, so asked across languages this returns every
    term of the text and the answer is about the language rather than about the
    corpus. `unfamiliar` returns the same terms with the share and a flag that
    says when that has happened; prefer it wherever the language of the text is
    not known to match the corpus.
    """
    from . import search as B

    df, _, _ = corpus_statistics(connection)
    seen: set[str] = set()
    out: list[str] = []
    for term in B.tokenize_text(B._normalize(text)):
        if not indexable_term(term) or term in df or term in seen:
            continue
        seen.add(term)
        out.append(term)
    return out


# There is no such thing as the share above which the list stops being readable,
# and the search for one went through two wrong shapes before that was measured.
#
# The first was exact equality -- every term unknown -- which is what a language
# crossing looks like and is not what it is. One shared word switched it off, and
# between related languages a cognate is always available: `color`, `radio`,
# `natural`, `total`, `error`, `region`, a number, an initialism or a proper noun
# all did it.
#
# The second was a threshold on the same quantity, 0.70, chosen inside a gap that
# looked wide: crossings at 0.7143 and above, same-language questions at 0.17 and
# below. Queries that sample did not contain closed the gap from both sides. A
# real crossing came in at 0.60 -- Spanish with two cognates -- and an
# English query against an English corpus reached 0.80, its unknown terms being
# four proper nouns. No single value is both below 0.60 and above 0.80.
#
# The reason is that the share measures how MUCH vocabulary is missing and never
# why. It goes high for two unrelated causes -- the corpus is in another
# language, or the corpus does not cover the subject -- and worse, it is not a
# property of the query at all: it is a property of the query and the package
# together, and it rises as the package shrinks. One query measured 0.17 against
# a package of 266 documents and 0.83 against one of 29, in the same language.
# A fixed cut on it therefore silences small packages systematically, which are
# exactly the ones for which "I have never seen these words" is the strongest
# thing they can say.
#
# So the language is decided by the language, and the share only says that
# something is missing at all. The detector is trusted when it speaks and not
# read as disagreement when it does not: failing to identify a language is not
# evidence of a different one. Measured 5 of 5 against 3 of 5 for the threshold.
#
# The residual risk is stated rather than guarded against, because guarding
# would mean inventing a constant nothing measured. Detection is wrong on short
# texts -- "how do you factor a quadratic polynomial" comes back Portuguese --
# and that case is saved here only because nothing in it is unknown. An English
# query misdetected as another language that also had a couple of unfamiliar
# words would be called a crossing wrongly.


def unfamiliar(connection: sqlite3.Connection, text: str) -> dict:
    """The unfamiliar part of a text, with what is needed to read it.

    `unknown_terms` answers literally and correctly, and a bare list of terms
    can be read as a measure of what the corpus does not know. Across languages
    it is not: the meaning index reaches a Spanish question against an English
    corpus, and the word index cannot, so every term comes back unknown. That is
    a measure of which language the corpus is written in, and nothing warns.

    So the share comes back with the terms, and a share of 1.0 -- everything
    unknown -- is the signature of a language crossing rather than of a strange
    question. `cross_language` says so outright when the corpus declares a
    language and the text looks like another, which is the promoted way to use
    mdcx and exactly where this signal saturates.
    """
    from . import search as B

    terms = [t for t in B.tokenize_text(B._normalize(text)) if indexable_term(t)]
    seen: set[str] = set()
    considered = [t for t in terms if not (t in seen or seen.add(t))]
    unknown = unknown_terms(connection, text)

    language = _corpus_language(connection)
    written_in, _ = B.detect_language(text)
    # Zero terms to consider is not a text made of familiar words. Left at zero
    # the share would read as "nothing unknown", which is the wrong answer to a
    # question the data cannot answer.
    share = len(unknown) / len(considered) if considered else None

    # The language decides the language, and the share only says that something
    # is missing at all. Both halves matter: without the share, a text misread
    # as another language would be called a crossing while sharing every word
    # with the corpus; without the language, no cut on the share separates, as
    # the account above it records.
    #
    # A detector that does not answer is not a detector that disagrees. Reading
    # its silence as difference was what silenced a small package answering an
    # English query against an English corpus -- there is no language in
    # "Moser spindle Hadwiger Nelson problem" to detect, and there was no
    # crossing either.
    crossed = bool(language and written_in and written_in != language
                   and unknown)
    return {
        "terms": unknown,
        "considered": len(considered),
        "share": round(share, 4) if share is not None else None,
        "language": language,
        "written_in": written_in,
        "cross_language": crossed,
        # What a consumer hangs a decision on. It is the crossing and not the
        # share, because a share is high for two unrelated reasons and only one
        # of them makes the list an artefact. A package that has never seen four
        # of five terms is making the strongest statement it can about the
        # question -- "I know nothing of this" -- and hanging this on the
        # fraction called exactly that unreadable.
        "meaningful": bool(unknown) and not crossed,
    }


def assess(connection: sqlite3.Connection, text: str) -> dict:
    """Everything this package can say about whether it is about a text.

    The two signals it holds disagree, and each alone is wrong in a way the
    other is not. Measured on a package of algebra: `answers` accepted "graph
    coloring adjacent vertices different colors" at 0.6553 against a threshold
    of 0.5661, and returned lessons on comparing graphs and on the ellipse --
    `graph` as the plot of a function, not as a graph. The word the question
    turns on, `coloring`, the corpus has never seen.

    That is not miscalibration and no quantity derived from the same vectors
    repairs it. Clearance was measured and does not separate: the false
    positive's clearance falls inside the range of the questions the corpus
    answers *and* inside the range of the unrelated ones, and its closeness sits
    above four of the eight legitimate questions. The minimum over windows of
    the query was measured and does not separate either. Both are functions of a
    space that has already lost the distinction: a multilingual embedding places
    the senses of a homonym together.

    What separates it lives in the other structure of the same package -- the
    literal vocabulary, which does tell an absent `coloring` from a present
    `graph`. This crosses the two and reports both; it does not overrule the
    verdict, because whether an unfamiliar word should refuse a query depends on
    what the query is for. A word may be peripheral: "the slope of a line drawn
    in Patagonia" is answerable and `patagonia` is unknown.
    """
    measured = closeness(connection, text)
    strange = unfamiliar(connection, text)
    return {
        "answers": answers(connection, text),
        "closeness": measured[0] if measured else None,
        "clearance": measured[1] if measured else None,
        "answerable_at": answerable_at(connection),
        "calibrated_from_questions": calibrated_from_questions(connection),
        "unknown_terms": strange["terms"],
        "unknown_share": strange["share"],
        # Suppressed where the list stops saying anything about the corpus:
        # across languages it is every term of the query, and a share this high
        # is unreadable whatever put it there.
        "unknown_terms_meaningful": strange["meaningful"],
        "cross_language": strange["cross_language"],
    }


def corpus_statistics(connection: sqlite3.Connection) -> tuple[dict, int, float]:
    """Document frequency per term, passage count and mean passage length.

    These feed BM25 directly, so reading them from another package does not
    fail: it reweights every term against a corpus the passages were not in.

    Public because weighting a term by how rare it is in a corpus is a
    reasonable thing to want, and the alternative was reading the `df` table
    over SQL -- which gives the counts without the rule that produced them. See
    `vocabulary`, which carries both.
    """
    return _corpus_statistics(connection)


# What a query needs from the corpus that is not a per-term count: how many
# passages it holds and how long they are on average. Two scalars, so they are
# read once and kept, where the frequencies themselves are asked for by term.
_SCALE_CACHE: dict = {}


def _has_token_counts(connection: sqlite3.Connection) -> bool:
    """Whether this package stored how long each passage is.

    Packages written before it did not, and are read by tokenising the
    candidates as before. The column is the fast path, not the only one.
    """
    key = _cache_key(connection)
    if key is not None and key in _TOKENS_CACHE:
        return _TOKENS_CACHE[key]
    try:
        with _CONNECTION_LOCK:
            columns = {row[1] for row in
                       connection.execute("PRAGMA table_info(passage)")}
            present = "tokens" in columns
            if present:
                row = connection.execute(
                    "SELECT count(*) FROM passage WHERE tokens IS NULL").fetchone()
                present = not row[0]
    except Exception:  # noqa: BLE001
        present = False
    if key is not None:
        _TOKENS_CACHE[key] = present
    return present


_TOKENS_CACHE: dict = {}

# The name of the vocabulary table this module creates on an open package to
# read term occurrences from the index. It holds no data of its own -- fts5vocab
# is a view over what FTS5 already stored -- so creating it costs nothing and
# writing it touches only the in-memory copy.
_INSTANCES = "mdcx_fts_instances"


# Whether this package's index tokenises the way this module does, decided per
# package by comparing them on real passages rather than by guessing from the
# script.
_AGREEMENT_CACHE: dict = {}

# How many passages to compare. Enough that a disagreement shows up, few enough
# that deciding costs nothing next to the query it saves.
_AGREEMENT_SAMPLE = 8


def _index_agrees(connection: sqlite3.Connection) -> bool:
    """Whether the terms FTS5 holds are the terms this module counts.

    They are not always. FTS5 tokenises with `unicode61`, and on Devanagari
    that splits words at the vowel marks: measured on one passage, 22 terms in
    the index against 14 from `tokenize_text`, with almost none in common. A
    count taken from the index there is a count of a different string, and the
    passage is dropped for holding none of the query -- which is what it looks
    like from outside: the language stops retrieving anything.

    Deciding by script would be guessing about a tokenizer this module does not
    own, so the two are compared on passages of this package. Once, and kept.
    """
    key = _cache_key(connection)
    if key is not None and key in _AGREEMENT_CACHE:
        return _AGREEMENT_CACHE[key]
    from . import search as B

    agrees = False
    try:
        with _CONNECTION_LOCK:
            connection.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {_INSTANCES} "
                "USING fts5vocab(passage_fts, instance)")
            rows = connection.execute(
                "SELECT id, text FROM passage ORDER BY id LIMIT ?",
                (_AGREEMENT_SAMPLE,)).fetchall()
            agrees = bool(rows)
            for identifier, text in rows:
                mine = {t for t in B.tokenize_text(B._normalize(text))}
                theirs = {t for (t,) in connection.execute(
                    f"SELECT DISTINCT term FROM {_INSTANCES} WHERE doc = ?",
                    (identifier,))}
                # Theirs may hold more -- the index also carries the title and
                # whatever else the indexed column joins -- but every term this
                # module would count has to be one the index knows under the
                # same name, or the count comes back wrong rather than absent.
                if not mine or not mine <= theirs:
                    agrees = False
                    break
    except Exception:  # noqa: BLE001 - an older or unusual package
        agrees = False
    if key is not None:
        _AGREEMENT_CACHE[key] = agrees
    return agrees


def _occurrences(connection: sqlite3.Connection, terms, rowids) -> dict:
    """How many times each term appears in each of these passages.

    Read from the index rather than by tokenising the passages again. FTS5
    already recorded every occurrence when the package was built; asking it is
    a lookup where re-reading the text is work proportional to the text.
    Measured over 1,200 candidate passages and four terms: 0.0057 s against
    0.1403 s, which is 24.7 times.

    It is also exact in a way a substring test is not -- it will not find
    `curve` inside `curvature` -- which is why the counts are kept at all.

    Returns None where the index cannot answer, so the caller tokenises. That
    covers more than an old package: what FTS5 holds is the *indexed* form of
    the text, and for a script that does not separate words -- Chinese,
    Japanese, Korean, and any term `segment_for_index` rewrites -- the token in
    the index is not the term that was asked for. Counting those from the index
    would silently answer about a different string, so the text is read
    instead. It is exactly the case the multilingual tests caught.
    """
    if not rowids or not terms:
        return {}
    if not _index_agrees(connection):
        return None
    try:
        with _CONNECTION_LOCK:
            connection.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {_INSTANCES} "
                "USING fts5vocab(passage_fts, instance)")
            found: dict = {}
            ids = list(rowids)
            for term in dict.fromkeys(terms):
                for start in range(0, len(ids), _TERMS_PER_STATEMENT):
                    batch = ids[start:start + _TERMS_PER_STATEMENT]
                    marks = ",".join("?" * len(batch))
                    for doc, how_many in connection.execute(
                            f"SELECT doc, count(*) FROM {_INSTANCES} "
                            f"WHERE term = ? AND doc IN ({marks}) GROUP BY doc",
                            [term] + batch):
                        found.setdefault(doc, {})[term] = how_many
            return found
    except Exception:  # noqa: BLE001 - an older or unusual package
        return None


def _corpus_scale(connection: sqlite3.Connection) -> tuple[int, float]:
    """How many passages the corpus holds, and their mean length."""
    key = _cache_key(connection)
    if key is not None and key in _SCALE_CACHE:
        return _SCALE_CACHE[key]
    with _CONNECTION_LOCK:
        row = connection.execute(
            "SELECT value FROM meta WHERE key='passages'").fetchone()
        n = int(json.loads(row[0])) if row else 1
        row = connection.execute(
            "SELECT value FROM meta WHERE key='mean_passage_length'").fetchone()
        lm = float(json.loads(row[0])) if row else 60.0
    if not n:
        n = 1
    if key is None:
        return n, lm
    _SCALE_CACHE[key] = (n, lm)
    return _SCALE_CACHE[key]


# How many terms to ask for in one statement. SQLite has a ceiling on host
# parameters, and a query widened by the shape sieve can carry more terms than
# anyone types.
_TERMS_PER_STATEMENT = 500


def _frequencies_for(connection: sqlite3.Connection,
                     terms) -> dict[str, int]:
    """How many passages hold each of these terms.

    Asked for by term rather than read whole. `df` has one row per term in the
    corpus -- measured at 521,231 rows on a corpus of mathematics -- and a
    query uses between one and ten of them, so reading the table made every
    query slower as the corpus grew, for counts it then discarded. The rows are
    keyed by term, so this is a lookup.
    """
    wanted = [t for t in dict.fromkeys(terms) if t]
    if not wanted:
        return {}
    found: dict[str, int] = {}
    with _CONNECTION_LOCK:
        for start in range(0, len(wanted), _TERMS_PER_STATEMENT):
            batch = wanted[start:start + _TERMS_PER_STATEMENT]
            placeholders = ",".join("?" * len(batch))
            for term, passages in connection.execute(
                    f"SELECT term, passages FROM df WHERE term IN ({placeholders})",
                    batch):
                found[term] = passages
    return found


def _corpus_statistics(connection: sqlite3.Connection) -> tuple[dict, int, float]:
    """Document frequency per term and mean passage length, as packed.

    The whole table, which is what `vocabulary()` publishes. A query does not
    go through here: it asks for the terms it has, through `_frequencies_for`.
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

        # Restored verbatim rather than rebuilt from passages, because there are
        # none: an attachment is stored whole. Leaving them out would make
        # export a lossy round trip, and silently -- the folder would look
        # complete.
        kept = 0
        for item in attachments(connection):
            relative = (item["pseudopath"][2:]
                        if item["pseudopath"].startswith("@/")
                        else item["pseudopath"])
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(item["text"], encoding="utf-8")
            kept += 1
    finally:
        connection.close()
    return {"documents": written, "target": str(target),
            **({"attachments": kept} if kept else {}),
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
    e.add_argument("--shapes", action="store_true",
                   help="build the index that recovers terms optical "
                        "recognition misread. It makes the package bigger -- "
                        "measured at 2.96%% on a corpus of books -- and buys "
                        "nothing on a corpus that never went through optical "
                        "recognition. `mdcx shapes` adds it to a package later")
    e.add_argument("--fast", action="store_true",
                   help="compress the package for speed rather than size. "
                        "Six times faster for about 38%% more bytes, measured. "
                        "For a package that is rewritten often rather than "
                        "distributed, where compressing is a fixed price paid "
                        "on every write")
    e.add_argument("--preset", type=int, metavar="0-9",
                   help="the LZMA level, overriding --fast. Sealing is a fixed "
                        "price paid on every write, and where a corpus is "
                        "rebuilt whenever it grows the clock costs more than "
                        "the bytes; a package meant to be distributed is the "
                        "opposite case. The level travels inside the stream, "
                        "so any reader opens what any level wrote")
    e.add_argument("--expressions", action="store_true",
                   help="index the expressions the word rule discards. A "
                        "lexical index is built out of words, and what makes "
                        "one formula differ from another is punctuation: "
                        "`b(b+p+q)` and `b(b-p-q)` reduce to the same terms, "
                        "and asked on their own to none at all. Worth it for a "
                        "corpus that is interrogated by statement, and nothing "
                        "for prose")
    e.add_argument("--quotes", choices=QUOTE_STRATEGIES, default="document",
                   help="how a quotation that runs past the end of a passage "
                        "is found. `document` keeps the whole normalised text "
                        "of every document, which has no length limit and is a "
                        "second copy of the corpus; `boundary` indexes the "
                        "join between consecutive passages, which finds a "
                        "quotation spanning one cut and not one longer than "
                        "the join")
    e.add_argument("--compression", choices=COMPRESSORS,
                   default=DEFAULT_COMPRESSION,
                   help="what compresses the body. The header records it and a "
                        "package is opened by what its header names, so this "
                        "changes nothing for existing packages. zstd opens "
                        "faster, which is what a server pays before it can "
                        "answer, and needs the `zstandard` module to read")
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

    n = sub.add_parser(
        "calibrate",
        help="measure a closed package against questions and store the result")
    n.add_argument("path")
    key_argument(n)
    n.add_argument("--question", action="append", required=True, metavar="QUESTION",
                   dest="questions",
                   help="a question this package is meant to answer. Repeat it. "
                        "The threshold goes just under the weakest of them, "
                        "which is the same rule pack --focus applies")
    n.add_argument("--signing-key", default="",
                   help="hex private key. Required if the package is signed: "
                        "rewriting it breaks the signature, and returning an "
                        "unsigned package where a signed one went in would be "
                        "a silent downgrade")

    h = sub.add_parser(
        "shapes",
        help="add the shape index to a package written without one")
    h.add_argument("path")
    key_argument(h)
    h.add_argument("--signing-key", default="",
                   help="hex private key. Required if the package is signed")

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
                 use_mtime=args.date_from_mtime, fast=args.fast,
                 shapes=args.shapes, preset=args.preset,
                 compression=args.compression, quotes=args.quotes,
                 expressions=args.expressions)
        print(f"Packed: {args.target}")
        print(f"  documents {r['documents']}   passages {r['passages']}"
              + (f"   attachments {r['attachments']}" if r.get("attachments") else ""))
        # Only when one document dominates. Printed always it would be noise;
        # printed at a third it is the thing worth knowing about the package.
        biggest = r.get("largest_document") or {}
        if biggest.get("share", 0) >= 0.33:
            print(f"  {biggest['name']} holds {100 * biggest['share']:.0f}% of the "
                  "passages")
        print(f"  database {r['bytes_database']:,} -> compressed {r['bytes_compressed']:,} "
              f"-> file {r['bytes_file']:,} bytes".replace(",", "."))
        print(f"  index {r['seconds_index']}s  seal {r['seconds_seal']}s")
        if r.get("embedding_model"):
            print(f"  meaning indexed with {r['embedding_model']} "
                  f"({r['embedding_dimensions']} dimensions)")
            if r.get("passages_reused"):
                print(f"  passages encoded {r['passages_encoded']:,}   "
                      f"reused {r['passages_reused']:,}".replace(",", "."))
        if r.get("shape_terms"):
            print(f"  shape index over {r['shape_terms']:,} term(s)".replace(",", "."))
        if r.get("answerable_at"):
            print(f"  answerable at {r['answerable_at']} "
                  f"(from {r['answerable_at_from']})"
                  + ("  <- inherited from the reused package"
                     if r.get("focus_inherited") else ""))
        return 0

    if args.action == "calibrate":
        r = calibrate(Path(args.path), resolve_key(args), args.questions,
                      args.signing_key)
        print(f"Calibrated: {args.path}")
        print(f"  answerable at {r['answerable_at']} (from {r['answerable_at_from']}), "
              f"measured against {r['questions']} question(s)")
        if not r["signed"]:
            print("  the package is not signed")
        return 0

    if args.action == "shapes":
        r = add_shapes(Path(args.path), resolve_key(args), args.signing_key)
        print(f"Shape index written: {args.path}")
        print(f"  {r['shape_terms']} term(s) of "
              f"{shapekey.MINIMUM_LENGTH} characters or more")
        if not r["signed"]:
            print("  the package is not signed")
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
        print(f"Content   : {header['documents']} documents, {header['passages']} passages"
              + (f", {header['attachments']} attachment(s)"
                 if header.get("attachments") else ""))
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
