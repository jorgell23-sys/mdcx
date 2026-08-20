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

# Maps a connection to the digest of the package it holds, so cached statistics
# are keyed by content rather than by id(), which Python recycles after garbage
# collection and could alias a stale entry from a closed package.
_CONNECTION_KEYS: dict[int, str] = {}

def _derive_key(key: str, salt: bytes) -> bytes:
    memory = 128 * SCRYPT_N * SCRYPT_R
    return hashlib.scrypt(key.encode("utf-8"), salt=salt,
                          n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=KEY_BYTES,
                          maxmem=memory * 2)

def _encrypt(datos: bytes, clave_derivada: bytes) -> tuple[bytes, bytes]:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    nonce = os.urandom(12)
    return nonce, AESGCM(clave_derivada).encrypt(nonce, datos, None)

def _decrypt(body: bytes, clave_derivada: bytes, nonce: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    return AESGCM(clave_derivada).decrypt(nonce, body, None)

def _build_database(folder: Path) -> tuple[bytes, dict]:
    """Build the in-memory database with documents, index and provenance."""
    from . import search as B

    docs = B.load_documents(folder)
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
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        -- Document frequency per term. FTS5 holds this internally but does not
        -- expose it usably, and without it the package cannot rank by the same
        -- criterion as the folder-based search.
        CREATE TABLE df (term TEXT PRIMARY KEY, passages INTEGER NOT NULL);
    """)

    n_passages = 0
    for i, d in enumerate(docs, 1):
        text = d["text"]
        archive = ""
        status = ""
        for line in text.splitlines()[:12]:
            if line.startswith("source_format:"):
                archive = line.split(":", 1)[1].strip()
            elif line.startswith("verification_status:"):
                status = line.split(":", 1)[1].strip()
        connection.execute(
            "INSERT INTO document VALUES (?,?,?,?,?,?,?,?)",
            (i, d["name"], d["pseudopath"], d["source"], d["folder"], archive, status,
             d["norm"]))
        for j, bloque in enumerate(d["blocks"] if "blocks" in d else _split_blocks(text)):
            if not bloque.strip():
                continue
            n_passages += 1
            # La columna indexada nunca queda vacia: FTS5 con contenido externo
            # lee esta columna directamente, y un nulo equivale a no indexar el
            # pasaje. Para texto que ya separa palabras es identica al original,
            # y la duplicacion la absorbe la compresion del paquete.
            connection.execute(
                "INSERT INTO passage VALUES (?,?,?,?,?)",
                (n_passages, i, j, bloque,
                 B._normalize(B.segment_for_index(bloque))))

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
    muestra = " ".join(d["text"][:4000] for d in docs[:40])
    idioma, confianza = B.detect_language(muestra)

    summary = {
        "documents": len(docs),
        "language": idioma,
        "language_confidence": round(confianza, 3),
        "passages": n_passages,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_folder": folder.name,
        "largo_medio_pasaje": round(avg_length, 2),
        "terminos_indexados": len(df_count),
    }
    manifiesto = folder / "_manifest.json"
    if manifiesto.exists():
        try:
            m = json.loads(manifiesto.read_text(encoding="utf-8"))
            summary["conversion"] = m.get("summary", {})
        except Exception:  # noqa: BLE001
            pass
    for k, v in summary.items():
        connection.execute("INSERT INTO meta VALUES (?,?)",
                    (k, json.dumps(v) if not isinstance(v, str) else v))
    connection.commit()

    datos = connection.serialize()
    connection.close()
    return bytes(datos), summary

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
         signing_key: str = "") -> dict:
    """Write the .mdcx file and return its figures."""
    import lzma

    if not folder.is_dir():
        raise ValueError(f"Not a folder: {folder}")

    t0 = time.perf_counter()
    base_score, summary = _build_database(folder)
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
    t_cifrado = time.perf_counter() - t0

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
        "seconds_encrypt": round(t_cifrado, 2),
        **summary,
    }

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
        marcadores = ",".join("?" * len(terms))
        fila = connection.execute(
            f"SELECT count(*) FROM df WHERE term IN ({marcadores})",
            tuple(terms)).fetchone()
    presentes = fila[0] if fila else 0
    if presentes:
        # Some term does exist in the corpus, so the empty result is about the
        # combination, not about the language.
        return None

    idioma = _corpus_language(connection)
    if not idioma:
        return ("None of the query terms appear in this corpus. If the documents "
                "are in another language, note that matching is literal and a "
                "query in a different language finds nothing.")

    consulta_idioma, _ = B.detect_language(query_text)
    if consulta_idioma and consulta_idioma == idioma:
        return None
    detalle = f" The query looks like '{consulta_idioma}'." if consulta_idioma else ""
    return (f"None of the query terms appear in this corpus, which is in "
            f"'{idioma}'.{detalle} Matching is literal, so a query in another "
            f"language finds nothing even when the material is present. Try the "
            f"same question in '{idioma}'.")


def _corpus_language(connection: sqlite3.Connection) -> str | None:
    """Language recorded when the package was built, if any."""
    with _CONNECTION_LOCK:
        fila = connection.execute(
            "SELECT value FROM meta WHERE key='language'").fetchone()
    if not fila or not fila[0]:
        return None
    try:
        valor = json.loads(fila[0])
    except Exception:  # noqa: BLE001
        valor = fila[0]
    return valor or None


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
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.deserialize(lzma.decompress(compressed))
    _CONNECTION_KEYS[id(connection)] = header["body_digest"]
    return connection, header

_SQL_TEMPLATE = """
    SELECT d.name, d.pseudopath, d.source, p.text, bm25(passage_fts) AS score
    FROM passage_fts
    JOIN passage p ON p.id = passage_fts.rowid
    JOIN document d ON d.id = p.{document_column}
    WHERE passage_fts MATCH ?
"""

def document_column(connection: sqlite3.Connection) -> str:
    """Name of the column linking a passage to its document.

    Packages written before 1.0.3 use a Spanish column name. Reading it from the
    table definition keeps those packages usable instead of failing on a name.
    """
    key = _CONNECTION_KEYS.get(id(connection), id(connection))
    cached = _COLUMN_CACHE.get(key)
    if cached:
        return cached
    with _CONNECTION_LOCK:
        names = [row[1] for row in connection.execute("PRAGMA table_info(passage)")]
    name = "document_id" if "document_id" in names else "documento_id"
    _COLUMN_CACHE[key] = name
    return name


def _run_match(connection: sqlite3.Connection, expr: str, limit: int,
              only: str | None) -> list[dict]:
    sql = _SQL_TEMPLATE.format(document_column=document_column(connection))
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
    return [{"document": r[0], "pseudopath": r[1], "source": r[2],
             "passage": r[3], "score": round(-r[4], 3)}
            for r in rows]

K1 = 1.5
B_LENGTH = 0.45
DOC_TOP_PASSAGES = 8

CANDIDATES = 1200

def query(connection: sqlite3.Connection, query_text: str, limit: int = 8,
              only: str | None = None) -> list[dict]:
    """Resolve a query, ranking by document rather than by isolated passage."""
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
        frec = _term_frequencies(r["passage"], distinct_terms)
        if not frec:
            continue
        largo = max(len(B.tokenize_text(B._normalize(r["passage"]))), 1)
        score = 0.0
        for t, f in frec.items():
            d_t = df.get(t, 1)
            idf = math.log(1 + (n_passages - d_t + 0.5) / (d_t + 0.5))
            score += idf * (f * (K1 + 1)) / (
                f + K1 * (1 - B_LENGTH + B_LENGTH * largo / avg_length))
        r = dict(r)
        r["score"] = round(score, 3)
        r["terms"] = sorted(frec)
        by_document.setdefault(r["document"], []).append(r)

    ranking = []
    for name, passages in by_document.items():
        passages.sort(key=lambda x: -x["score"])
        base_score = sum(x["score"] for x in passages[:DOC_TOP_PASSAGES])
        coverage = max(len(x["terms"]) for x in passages) / max(len(distinct_terms), 1)
        ranking.append((base_score * coverage, passages))
    ranking.sort(key=lambda par: -par[0])

    if len(phrase.split()) >= 5:
        needle = B._normalize(phrase)
        with _CONNECTION_LOCK:
            preferred = [n for (n, t) in connection.execute(
                "SELECT name, normalized_text FROM document") if t and needle in t]
        if preferred:
            position = {d: i for i, d in enumerate(preferred)}
            ranking.sort(key=lambda par: (position.get(par[1][0]["document"], len(position)),
                                          -par[0]))

    out: list[dict] = []
    for round_index in range(DOC_TOP_PASSAGES):
        for score, passages in ranking:
            if round_index < len(passages):
                r = dict(passages[round_index])
                r["score_documento"] = round(score, 3)
                out.append(r)
                if len(out) >= limit:
                    return out
    return out[:limit]

def _term_frequencies(text: str, terms: set[str]) -> dict[str, int]:
    from . import search as B

    cuenta: dict[str, int] = {}
    for t in B.tokenize_text(B._normalize(text)):
        if t in terms:
            cuenta[t] = cuenta.get(t, 0) + 1
    return cuenta

_STATS_CACHE: dict = {}

# Resolved column name per package, so the lookup happens once.
_COLUMN_CACHE: dict = {}

def _corpus_statistics(connection: sqlite3.Connection) -> tuple[dict, int, float]:
    """Document frequency per term and mean passage length, as packed."""
    key = _CONNECTION_KEYS.get(id(connection), id(connection))
    if key not in _STATS_CACHE:
        with _CONNECTION_LOCK:
            df = {t: n for t, n in connection.execute("SELECT term, passages FROM df")}
            row = connection.execute(
                "SELECT value FROM meta WHERE key='passages'").fetchone()
        n = int(json.loads(row[0])) if row else max(len(df), 1)
        row = connection.execute(
            "SELECT value FROM meta WHERE key='largo_medio_pasaje'").fetchone()
        lm = float(json.loads(row[0])) if row else 60.0
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
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="The .mdcx format: an indexed, encrypted, portable corpus")
    sub = ap.add_subparsers(dest="action", required=True)

    e = sub.add_parser("pack")
    e.add_argument("--output", default="Output")
    e.add_argument("--target", default="corpus.mdcx")
    e.add_argument("--key", required=True)
    e.add_argument("--issuer", default="")
    e.add_argument("--signing-key", default="",
                   help="hex private key to sign the package with")

    k = sub.add_parser("keygen")

    v = sub.add_parser("verify")
    v.add_argument("path")
    v.add_argument("--public-key", required=True)

    i = sub.add_parser("info")
    i.add_argument("path")

    x = sub.add_parser("export")
    x.add_argument("path")
    x.add_argument("--target", required=True)
    x.add_argument("--key", required=True)

    b = sub.add_parser("search")
    b.add_argument("path")
    b.add_argument("query_text")
    b.add_argument("--key", required=True)
    b.add_argument("--limit", type=int, default=5)
    b.add_argument("--only", choices=["received", "sent"])

    args = ap.parse_args()

    if args.action == "pack":
        r = pack(Path(args.output), Path(args.target), args.key, args.issuer,
                 args.signing_key)
        print(f"Packed: {args.target}")
        print(f"  documents {r['documents']}   passages {r['passages']}")
        print(f"  database {r['bytes_database']:,} -> compressed {r['bytes_compressed']:,} "
              f"-> file {r['bytes_file']:,} bytes".replace(",", "."))
        print(f"  index {r['seconds_index']}s  compress {r['seconds_compress']}s  "
              f"encrypt {r['seconds_encrypt']}s")
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
        if header.get("conversion"):
            print(f"Conversion: {json.dumps(header['conversion'], ensure_ascii=False)[:200]}")
        return 0

    if args.action == "export":
        r = export(Path(args.path), args.key, Path(args.target))
        print(f"Exported {r['documents']} documents to {r['target']}")
        return 0

    connection, header = open_package(Path(args.path), args.key)
    results = query(connection, args.query_text, args.limit, args.only)
    print(f"{len(results)} passage(s)\n")
    for r in results:
        print("-" * 96)
        print(f"[{r['source']}] {r['document']}   [score {r['score']}]")
        print(f"{r['pseudopath']}")
        print(r["passage"][:1200])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
