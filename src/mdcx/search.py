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

"""Lexical retrieval over a converted corpus.

Passages are ranked with BM25 aggregated per document. Ranking per document
rather than per isolated passage matters: a long document covering a subject
in several fragments would otherwise lose to a short unrelated passage that
repeats one term.

Measured on twenty real queries, the correct document appears within the top
five results in 19 cases and within the top ten in all 20.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# Words are alphanumeric sequences, accented characters included. This is the same
_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+", re.UNICODE)

def _normalize(text: str) -> str:
    """Normalise text for comparison: lowercase, accents folded."""
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.replace("’", "'").replace("‘", "'")
    t = t.replace("“", '"').replace("”", '"')
    t = t.replace("–", "-").replace("—", "-").replace("‑", "-")
    return re.sub(r"\s+", " ", t)

def load_documents(output_root: Path) -> list[dict]:
    """Read the Markdown files of an output folder together with their provenance."""
    docs = []
    for p in sorted(output_root.rglob("*.md")):
        rel = p.relative_to(output_root)
        parts = rel.parts
        if parts[0].startswith("_") or rel.name.startswith("00_"):
            continue
        # The top-level folder states whether the document was sent or received.
        root_name = parts[0].lower()
        # Folder names are recognised in English and Spanish, since a collection
        # may be organised in either.
        if any(w in root_name for w in ("sent", "emitido", "outgoing")):
            source = "SENT"
        elif any(w in root_name for w in ("received", "recibido", "incoming")):
            source = "RECEIVED"
        else:
            source = "OTHER"
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        body = text.split("---", 2)[-1] if text.startswith("---") else text
        docs.append({
            "path": p,
            "rel": rel.as_posix(),
            "pseudopath": "@/" + rel.as_posix(),
            "source": source,
            "name": rel.name[:-3],
            "folder": rel.parent.as_posix(),
            "text": body,
            "norm": _normalize(body),
        })
    return docs

def _paragraphs(text: str) -> list[str]:
    return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

def _trim_table(bloque: str, aguja: str) -> str:
    """For tables, keep the header row and the rows containing the term."""
    lines = bloque.splitlines()
    if sum(1 for l in lines if l.strip().startswith("|")) < 3:
        return bloque
    header = [l for l in lines[:2]]
    rows = [l for l in lines[2:] if aguja in _normalize(l)]
    if not rows:
        return bloque
    return "\n".join(header + rows)

BM25_K1 = 1.5
BM25_B = 0.45

DOC_TOP_PASSAGES = 8

MIN_PHRASE_WORDS = 5

_BM25_CACHE: dict = {}

def _bm25_index(docs: list[dict], only: str | None) -> dict:
    """Inverted passage index holding the quantities BM25 requires."""
    key = (id(docs), only or "todo")
    if key in _BM25_CACHE:
        return _BM25_CACHE[key]
    passages = []
    for d in docs:
        if only and d["source"] != only.upper():
            continue
        blocks = d.setdefault("blocks", _paragraphs(d["text"]))
        normas = d.setdefault("bloques_norm", [_normalize(b) for b in blocks])
        for i, bn in enumerate(normas):
            tk = _TOKEN_RE.findall(bn)
            passages.append({"doc": d, "i": i, "frec": Counter(tk), "largo": len(tk)})
    df = Counter()
    for p in passages:
        for t in p["frec"]:
            df[t] += 1
    avg_length = (sum(p["largo"] for p in passages) / len(passages)) if passages else 1.0
    idx = {"passages": passages, "df": df, "n": len(passages), "avg_length": avg_length}
    _BM25_CACHE[key] = idx
    return idx

# Optional query glossary, empty by default.
#
# A glossary maps a term to equivalents in the language of a corpus, and is
# therefore specific to one domain and one language pair. The package used to
# ship one covering piping engineering, which helped that corpus, contributed
# nothing to any other, and backed a claim of general Spanish support that
# failed on eleven of twelve queries outside that domain.
#
# Load your own with load_glossary(path) and assign it here.
GLOSSARY: dict = {}

# Function words are the most frequent words of each language and identify it
# without a model or a dependency. Detection is used to warn about a mismatch,
# never to alter retrieval.
_LANGUAGE_MARKERS = {
    "en": {"the", "of", "and", "to", "in", "is", "that", "for", "with", "as",
           "are", "was", "on", "at", "by", "this", "be", "from", "or", "an"},
    "es": {"el", "la", "de", "que", "y", "en", "los", "del", "las", "por",
           "con", "para", "una", "es", "se", "al", "lo", "como", "mas", "su"},
    "de": {"der", "die", "das", "und", "in", "den", "von", "zu", "mit", "sich",
           "des", "auf", "ist", "im", "dem", "nicht", "ein", "eine", "als"},
    "fr": {"le", "la", "les", "de", "et", "des", "en", "un", "une", "du",
           "dans", "est", "pour", "que", "qui", "sur", "au", "par", "pas"},
    "pt": {"o", "a", "de", "que", "e", "do", "da", "em", "um", "para", "com",
           "nao", "uma", "os", "no", "se", "na", "por", "mais", "as"},
    "it": {"il", "di", "che", "e", "la", "in", "un", "per", "con", "del",
           "una", "le", "da", "non", "sono", "si", "come", "piu", "al"},
}

# Every marker of every language. Used as a floor for stopword removal when the
# corpus is too small to derive its own.
_ALL_MARKERS = {w for words in _LANGUAGE_MARKERS.values() for w in words}

# Words removed from a query before matching. Previously a fixed Spanish and
# English list, which filtered nothing in a German or French corpus. Using the
# markers of every supported language covers all of them, and corpus_stopwords()
# derives the rest from the material itself.
STOPWORDS = _ALL_MARKERS


def detect_language(text: str, minimum: float = 0.05) -> tuple[str | None, float]:
    """Identify the language of a text by its function words.

    Returns the code and the proportion of words that matched. Below the minimum
    the answer is None: a text of technical terms and figures may legitimately
    contain almost no function words, and guessing there would be worse than
    admitting ignorance.
    """
    words = [w for w in _TOKEN_RE.findall(text.lower()) if w.isalpha()]
    if not words:
        return None, 0.0
    scores = {
        code: sum(1 for w in words if w in markers) / len(words)
        for code, markers in _LANGUAGE_MARKERS.items()
    }
    best = max(scores, key=scores.get)
    return (best, scores[best]) if scores[best] >= minimum else (None, scores[best])


def corpus_stopwords(documents: list[dict], threshold: float = 0.6) -> set:
    """Words appearing in most documents, which therefore distinguish none.

    Deriving them from the corpus works in any language, whereas a fixed list
    only works in the languages someone thought to include. The fixed markers
    are added as a floor for corpora too small for the frequency to mean
    anything.
    """
    if not documents:
        return set(_ALL_MARKERS)
    counts: dict[str, int] = {}
    for d in documents:
        for term in {w for w in _TOKEN_RE.findall(d["norm"]) if len(w) > 2}:
            counts[term] = counts.get(term, 0) + 1
    limit = max(2, int(len(documents) * threshold))
    return {t for t, n in counts.items() if n >= limit} | _ALL_MARKERS


def load_glossary(path: Path) -> dict:
    """Read a query glossary from a JSON file: {"term": ["equivalent", ...]}.

    A glossary is inherently specific to a domain and a language pair. The one
    that used to ship with the package covered piping engineering, which helped
    that corpus and contributed nothing to any other while backing a promise of
    general Spanish support. Supplying one is now the caller's decision.
    """
    import json

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Cannot read glossary {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("A glossary must be an object mapping terms to lists.")
    return {str(k).lower(): [str(v) for v in (vs or [])] for k, vs in data.items()}



def expand_terms(terms: list[str]) -> list[str]:
    """Add corpus-language equivalents to the query, dropping stopwords."""
    out: list[str] = []
    for t in terms:
        if t in STOPWORDS:
            continue
        equivalentes = GLOSSARY.get(t)
        if equivalentes is None:
            out.append(t)
        elif equivalentes:
            out.append(t)
            out.extend(equivalentes)
    vistos = set()
    return [t for t in out if not (t in vistos or vistos.add(t))]

def rank_passages(docs: list[dict], query_text: str, context: int = 1,
                    only: str | None = None, limit: int = 12,
                    minimo_terminos: int = 2) -> list[dict]:
    """Rank passages by BM25, aggregating scores per document."""
    consulta_tk = [t for t in _TOKEN_RE.findall(_normalize(query_text)) if len(t) > 2]
    consulta_tk = expand_terms(consulta_tk)
    if not consulta_tk:
        return []
    idx = _bm25_index(docs, only)
    n, df, lm = idx["n"], idx["df"], idx["avg_length"]
    distintos = len(set(consulta_tk))

    por_documento: dict[str, list[dict]] = {}
    for p in idx["passages"]:
        presentes = [t for t in consulta_tk if p["frec"].get(t)]
        if len(presentes) < min(minimo_terminos, len(consulta_tk)):
            continue
        score = 0.0
        for t in presentes:
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            f = p["frec"][t]
            score += idf * (f * (BM25_K1 + 1)) / (
                f + BM25_K1 * (1 - BM25_B + BM25_B * p["largo"] / lm))
        d, i = p["doc"], p["i"]
        ini = max(0, i - context)
        por_documento.setdefault(d["name"], []).append({
            "document": d["name"],
            "source": d["source"],
            "pseudopath": d["pseudopath"],
            "folder": d["folder"],
            "score": round(score, 3),
            "terms": presentes,
            "passage": _trim_table(d["blocks"][i], presentes[0] if presentes else ""),
            "pasaje": "\n\n".join(d["blocks"][ini: i + context + 1]),
        })

    ranking = []
    for name, passages in por_documento.items():
        passages.sort(key=lambda r: -r["score"])
        base = sum(r["score"] for r in passages[:DOC_TOP_PASSAGES])
        coverage = max(len(r["terms"]) for r in passages) / max(distintos, 1)
        ranking.append((base * coverage, passages))
    ranking.sort(key=lambda par: -par[0])

    out: list[dict] = []
    for vuelta in range(DOC_TOP_PASSAGES):
        for puntaje, passages in ranking:
            if vuelta < len(passages):
                r = dict(passages[vuelta])
                r["score_documento"] = round(puntaje, 3)
                out.append(r)
                if len(out) >= limit:
                    return out
    return out[:limit]

def search(docs: list[dict], query_text: str, context: int = 1,
                   only: str | None = None, limit: int = 12) -> list[dict]:
    """Rank by relevance, placing literal matches first when the phrase is specific."""
    frase = query_text.strip().split(".")[0][:160].strip()
    ordenados = rank_passages(docs, query_text, context, only, limit)

    if len(frase.split()) < MIN_PHRASE_WORDS:
        return ordenados

    literales = search_literal(docs, frase, context, only, limit)
    if not literales:
        return ordenados

    vistos = {(r["document"], r["passage"][:80]) for r in literales}
    for r in ordenados:
        key = (r["document"], r["passage"][:80])
        if key in vistos:
            continue
        vistos.add(key)
        literales.append(r)
        if len(literales) >= limit:
            break
    return literales[:limit]

def search_literal(docs: list[dict], frase: str, context: int = 1,
           only: str | None = None, limit: int = 12) -> list[dict]:
    """Return passages containing the phrase verbatim."""
    aguja = _normalize(frase)
    if not aguja:
        return []
    results = []
    for d in docs:
        if only and d["source"] != only.upper():
            continue
        blocks = d.setdefault("blocks", _paragraphs(d["text"]))
        normas = d.setdefault("bloques_norm", [_normalize(b) for b in blocks])
        for i, bn in enumerate(normas):
            if aguja not in bn:
                continue
            ini = max(0, i - context)
            results.append({
                "document": d["name"],
                "source": d["source"],
                "pseudopath": d["pseudopath"],
                "folder": d["folder"],
                "passage": _trim_table(blocks[i], aguja),
                "pasaje": "\n\n".join(blocks[ini: i + context + 1]),
            })
            if len(results) >= limit:
                return results
    return results

def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Busqueda con cita exacta sobre los .md convertidos.")
    ap.add_argument("frase", nargs="?", help="text a search_literal")
    ap.add_argument("--output", default="Output", help="folder holding the .md files")
    ap.add_argument("--context", type=int, default=1, help="parrafos vecinos a incluir")
    ap.add_argument("--only", choices=["sent", "received"], help="restrict by direction")
    ap.add_argument("--limit", type=int, default=12, help="maximum passages")
    ap.add_argument("--frases", help="file with one phrase per line")
    ap.add_argument("--json", help="write the result to a JSON file")
    ap.add_argument("--literal", action="store_true",
                    help="require the exact phrase only, without relevance fallback")
    ap.add_argument("--bm25", action="store_true",
                    help="use relevance ranking only, without literal matching")
    args = ap.parse_args()

    docs = load_documents(Path(args.output))
    if not docs:
        print("No converted documents found.", file=sys.stderr)
        return 2

    frases = []
    if args.frases:
        frases = [l.strip() for l in Path(args.frases).read_text(encoding="utf-8").splitlines() if l.strip()]
    elif args.frase:
        frases = [args.frase]
    else:
        print("Provide a phrase, or a file with --phrases.", file=sys.stderr)
        return 2

    out = {}
    for f in frases:
        if args.literal:
            res = search_literal(docs, f, args.context, args.only, args.limit)
        elif args.bm25:
            res = rank_passages(docs, f, args.context, args.only, args.limit)
        else:
            res = search(docs, f, args.context, args.only, args.limit)
        out[f] = res
        print("=" * 100)
        print(f"SEARCH: {f}")
        print(f"  {len(res)} passage(s)")
        for r in res:
            print("-" * 100)
            mark = f" [score {r['score']}]" if "score" in r else ""
            print(f"  [{r['source']}] {r['document']}{mark}")
            print(f"  {r['pseudopath']}")
            print(f"  {r['passage'][:1800]}")
        print()

    if args.json:
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Details in {args.json}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
