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

from . import console


def _combining_mark_ranges() -> str:
    r"""Character ranges of every combining mark, read from the Unicode database.

    A word is not only letters. In Devanagari, Bengali, Tamil, Thai and others
    the vowels are written as combining marks, and Python excludes marks from
    \w: a pattern built on \w splits those words at every vowel, leaving
    fragments that match nothing. The ranges are derived rather than listed, so
    no script is covered because someone remembered it.
    """
    ranges: list[tuple[int, int]] = []
    start = previous = None
    for code in range(0x300, 0x1F000):
        if unicodedata.category(chr(code))[0] == "M":
            if start is None:
                start = code
            previous = code
        elif start is not None:
            ranges.append((start, previous))
            start = None
    if start is not None:
        ranges.append((start, previous))
    return "".join(
        re.escape(chr(low)) if low == high
        else f"{re.escape(chr(low))}-{re.escape(chr(high))}"
        for low, high in ranges)


# Words are runs of letters, digits and the marks that belong to them, in any
# writing system.
_TOKEN_RE = re.compile(r"(?:[^\W_]|[" + _combining_mark_ranges() + r"])+", re.UNICODE)

# Writing systems that do not separate words with spaces. A run of their
# characters is a sentence rather than a word, so a lexical index has nothing to
# match unless the run is split first. Splitting on the character is what can be
# done without a segmenter trained on one particular language, which would cover
# that language and leave the rest exactly where they started.
_UNSPACED_RANGES = (
    (0x0E00, 0x0E7F),    # Thai
    (0x0E80, 0x0EFF),    # Lao
    (0x0F00, 0x0FFF),    # Tibetan
    (0x1000, 0x109F),    # Myanmar
    (0x1780, 0x17FF),    # Khmer
    (0x3040, 0x30FF),    # kana
    (0x3400, 0x4DBF),    # CJK extension A
    (0x4E00, 0x9FFF),    # CJK unified ideographs
    (0xA980, 0xA9DF),    # Javanese
    (0xAC00, 0xD7AF),    # hangul syllables
    (0xF900, 0xFAFF),    # compatibility ideographs
)


_UNSPACED_RE = re.compile(
    "[" + "".join(f"{re.escape(chr(low))}-{re.escape(chr(high))}"
                  for low, high in _UNSPACED_RANGES) + "]")


def _has_unspaced(text: str) -> bool:
    """Whether a text holds any character of a writing system without spaces.

    Segmentation is only needed for those, and a corpus in a spaced script is
    the common case: the test has to be a compiled search rather than a loop
    over every character.
    """
    return _UNSPACED_RE.search(text) is not None


def _is_unspaced(character: str) -> bool:
    """True for a character of a writing system that does not use spaces."""
    code = ord(character)
    return any(low <= code <= high for low, high in _UNSPACED_RANGES)


# Retained under its former name for the call sites that read as "one character
# is one word".
_is_cjk = _is_unspaced


def tokenize_text(text: str) -> list[str]:
    """Split text into searchable units, in any writing system.

    Runs of CJK characters are split into individual characters; everything else
    is split on non-word boundaries.
    """
    runs = _TOKEN_RE.findall(text)
    if not _has_unspaced(text):
        return runs
    out: list[str] = []
    for run in runs:
        if _has_unspaced(run):
            out.extend(c for c in run if not c.isspace())
        else:
            out.append(run)
    return out



def segment_for_index(text: str) -> str:
    """Insert spaces between CJK characters so a lexical index can match them.

    Scripts that already separate words are returned unchanged, so this costs
    nothing outside Chinese, Japanese and Korean.
    """
    if not _has_unspaced(text):
        return text
    out = []
    previous_cjk = False
    for c in text:
        current_cjk = _is_cjk(c)
        if current_cjk and out and not out[-1].isspace():
            out.append(" ")
        elif previous_cjk and not current_cjk and not c.isspace():
            out.append(" ")
        out.append(c)
        previous_cjk = current_cjk
    return "".join(out)


def searchable_terms(text: str, minimum: int = 3) -> list[str]:
    """Tokens worth searching for, in any writing system.

    The minimum length filters noise in scripts that separate words. It cannot
    apply to CJK, where one character is a whole word: a length rule written for
    Latin text would discard every Chinese, Japanese and Korean term.
    """
    return [t for t in tokenize_text(text)
            if len(t) >= minimum or (len(t) == 1 and _is_cjk(t))]


# Combining marks that represent an accent placed on an alphabetic letter. These
# are what folding is meant to remove, so that "cafe" matches "cafÃ©". Every other
# combining mark belongs to the letter it sits on: the vowel signs of Devanagari,
# Bengali, Tamil and Thai, and the points of Hebrew and Arabic, are written as
# combining characters but carry the sound of the syllable. Removing those does
# not fold a word, it destroys it.
_FOLDABLE_MARKS = (
    (0x0300, 0x036F),    # combining diacritical marks
    (0x1AB0, 0x1AFF),    # extended
    (0x1DC0, 0x1DFF),    # supplement
    (0x20D0, 0x20F0),    # for symbols
    (0xFE20, 0xFE2F),    # half marks
)


# Built once as a translation table. Folding runs over every passage of a corpus,
# so the test has to happen in the string method rather than in a Python call per
# character.
_FOLD_TABLE = {code: None
               for low, high in _FOLDABLE_MARKS
               for code in range(low, high + 1)
               if unicodedata.combining(chr(code))}


def _is_foldable_mark(character: str) -> bool:
    """True for a combining mark that folding is meant to remove."""
    return ord(character) in _FOLD_TABLE


def _normalize(text: str) -> str:
    """Normalise text for comparison: lowercase, accents folded."""
    t = unicodedata.normalize("NFKD", text.lower()).translate(_FOLD_TABLE)
    t = t.replace("’", "'").replace("‘", "'")
    t = t.replace("“", '"').replace("”", '"')
    t = t.replace("–", "-").replace("—", "-").replace("‑", "-")
    # Recompose. Decomposition splits Hangul syllables into individual jamo,
    # which no index built from syllables can match; composing them back leaves
    # every other script as it was, since their marks were dropped above.
    t = unicodedata.normalize("NFC", t)
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

def _trim_table(block: str, needle: str) -> str:
    """For tables, keep the header row and the rows containing the term."""
    lines = block.splitlines()
    if sum(1 for l in lines if l.strip().startswith("|")) < 3:
        return block
    header = [l for l in lines[:2]]
    rows = [l for l in lines[2:] if needle in _normalize(l)]
    if not rows:
        return block
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
        rules = d.setdefault("normalised_blocks", [_normalize(b) for b in blocks])
        for i, bn in enumerate(rules):
            tk = tokenize_text(bn)
            passages.append({"doc": d, "i": i, "freq": Counter(tk), "length": len(tk)})
    df = Counter()
    for p in passages:
        for t in p["freq"]:
            df[t] += 1
    avg_length = (sum(p["length"] for p in passages) / len(passages)) if passages else 1.0
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
        for term in set(searchable_terms(d["norm"])):
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



def expand_terms(terms: list[str], language: str | None = None) -> list[str]:
    """Add corpus-language equivalents to the query, dropping stopwords."""
    # Only the function words of the corpus language are removed. Removing those
    # of six languages at once cost real terms: "die" in die casting, "das" in
    # Das Kapital, "les" in Les Miserables all carry meaning in the language
    # being searched.
    stopwords = _LANGUAGE_MARKERS.get(language, set()) if language else STOPWORDS

    out: list[str] = []
    for t in terms:
        if t in stopwords:
            continue
        equivalents = GLOSSARY.get(t)
        if equivalents is None:
            out.append(t)
        elif equivalents:
            out.append(t)
            out.extend(equivalents)
    seen = set()
    result = [t for t in out if not (t in seen or seen.add(t))]
    # Filtering must never leave nothing to search for. "The Who" is a band and
    # "in vitro" is a technique; if every term was a function word, the query is
    # about those words.
    if not result:
        return [t for t in terms if not (t in seen or seen.add(t))]
    return result

def rank_passages(docs: list[dict], query_text: str, context: int = 1,
                    only: str | None = None, limit: int = 12,
                    min_terms: int = 2) -> list[dict]:
    """Rank passages by BM25, aggregating scores per document."""
    query_tokens = searchable_terms(_normalize(query_text))
    query_tokens = expand_terms(query_tokens)
    if not query_tokens:
        return []
    idx = _bm25_index(docs, only)
    n, df, lm = idx["n"], idx["df"], idx["avg_length"]
    distinct = len(set(query_tokens))

    by_document: dict[str, list[dict]] = {}
    for p in idx["passages"]:
        present = [t for t in query_tokens if p["freq"].get(t)]
        if len(present) < min(min_terms, len(query_tokens)):
            continue
        score = 0.0
        for t in present:
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            f = p["freq"][t]
            score += idf * (f * (BM25_K1 + 1)) / (
                f + BM25_K1 * (1 - BM25_B + BM25_B * p["length"] / lm))
        d, i = p["doc"], p["i"]
        ini = max(0, i - context)
        by_document.setdefault(d["name"], []).append({
            "document": d["name"],
            "source": d["source"],
            "pseudopath": d["pseudopath"],
            "folder": d["folder"],
            "score": round(score, 3),
            "terms": present,
            "passage": _trim_table(d["blocks"][i], present[0] if present else ""),
            "pasaje": "\n\n".join(d["blocks"][ini: i + context + 1]),
        })

    ranking = []
    for name, passages in by_document.items():
        passages.sort(key=lambda r: -r["score"])
        base = sum(r["score"] for r in passages[:DOC_TOP_PASSAGES])
        coverage = max(len(r["terms"]) for r in passages) / max(distinct, 1)
        ranking.append((base * coverage, passages))
    ranking.sort(key=lambda pair: -pair[0])

    out: list[dict] = []
    for round_no in range(DOC_TOP_PASSAGES):
        for doc_score, passages in ranking:
            if round_no < len(passages):
                r = dict(passages[round_no])
                r["document_score"] = round(doc_score, 3)
                out.append(r)
                if len(out) >= limit:
                    return out
    return out[:limit]

def search(docs: list[dict], query_text: str, context: int = 1,
                   only: str | None = None, limit: int = 12) -> list[dict]:
    """Rank by relevance, placing literal matches first when the phrase is specific."""
    phrase = query_text.strip().split(".")[0][:160].strip()
    ordered = rank_passages(docs, query_text, context, only, limit)

    if len(phrase.split()) < MIN_PHRASE_WORDS:
        return ordered

    literals = search_literal(docs, phrase, context, only, limit)
    if not literals:
        return ordered

    seen = {(r["document"], r["passage"][:80]) for r in literals}
    for r in ordered:
        key = (r["document"], r["passage"][:80])
        if key in seen:
            continue
        seen.add(key)
        literals.append(r)
        if len(literals) >= limit:
            break
    return literals[:limit]

def search_literal(docs: list[dict], phrase: str, context: int = 1,
           only: str | None = None, limit: int = 12) -> list[dict]:
    """Return passages containing the phrase verbatim."""
    needle = _normalize(phrase)
    if not needle:
        return []
    results = []
    for d in docs:
        if only and d["source"] != only.upper():
            continue
        blocks = d.setdefault("blocks", _paragraphs(d["text"]))
        rules = d.setdefault("normalised_blocks", [_normalize(b) for b in blocks])
        for i, bn in enumerate(rules):
            if needle not in bn:
                continue
            ini = max(0, i - context)
            results.append({
                "document": d["name"],
                "source": d["source"],
                "pseudopath": d["pseudopath"],
                "folder": d["folder"],
                "passage": _trim_table(blocks[i], needle),
                "pasaje": "\n\n".join(blocks[ini: i + context + 1]),
            })
            if len(results) >= limit:
                return results
    return results

def main() -> int:
    console.configure()

    ap = argparse.ArgumentParser(
        description="Search the converted .md files, quoting each passage exactly.")
    from . import __version__
    ap.add_argument("--version", action="version", version=f"mdcx {__version__}")
    ap.add_argument("phrase", metavar="PHRASE", nargs="?",
                    help="the phrase or question to search for")
    ap.add_argument("--output", default="Output", help="folder holding the .md files")
    ap.add_argument("--context", type=int, default=1,
                    help="neighbouring paragraphs to include with each passage")
    ap.add_argument("--only", choices=["sent", "received"], help="restrict by direction")
    ap.add_argument("--limit", type=int, default=12, help="maximum passages")
    # --frases was the original spelling and is kept so a script written against
    # it keeps working; the error message already pointed at --phrases, which is
    # the name a reader of the help would look for.
    ap.add_argument("--phrases", "--frases", dest="phrases", metavar="FILE",
                    help="file with one phrase per line")
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

    phrases = []
    if args.phrases:
        phrases = [l.strip() for l in Path(args.phrases).read_text(encoding="utf-8").splitlines() if l.strip()]
    elif args.phrase:
        phrases = [args.phrase]
    else:
        print("Provide a phrase, or a file with --phrases.", file=sys.stderr)
        return 2

    out = {}
    for f in phrases:
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
