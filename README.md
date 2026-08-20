# mdcx

<!-- mcp-name: io.github.jorgell23-sys/markdown-document-search -->

[![PyPI](https://img.shields.io/pypi/v/mdcx)](https://pypi.org/project/mdcx/) [![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22015991.svg)](https://doi.org/10.5281/zenodo.22015991)

Convert a document collection to verified Markdown, package it into a single
encrypted file, and make it queryable by agents through the Model Context
Protocol.

## The problem

An agent answering questions about a document collection has two options. It can
receive the documents in its context window, which is expensive and bounded by
the window size. Or it can query a component that already knows where each item
is.

Measuring one specific query — where the minimum pipe diameter to be modelled in
3D is stated — over a real collection of 99 documents and 180 MB, using the
cl100k_base tokenizer:

| | Model tokens | Local tokens |
|---|---|---|
| Reading the originals | **2,265,488** | 2,265,327 |
| Querying the package | **435** | 2,688,861 |

The 435 comprise 20 for the question, 274 for the retrieved passage and 141 for
the answer.

The first row costs the entire collection for a concrete reason: a PDF cannot be
searched, it is a binary, and without prior conversion there is no way to know
which of the 99 documents holds the answer. They all have to be extracted and
read.

This is one measurement, not an average: the saving depends on how much text an
answer requires. What does not vary is the shape of the change. The work does not
disappear, it moves from the context window — which is billed and finite — to the
CPU, which is not. That is why the local column rises rather than falls.

## The three stages

**Conversion.** Each document is converted to Markdown and checked against the
text the original actually exposes, read with a library independent from the
engine that performed the conversion. Content the structured engine omits is
appended verbatim rather than reported as lost.

Over the collection used during development — 99 documents, 1,144,553 reference
words — 594 words were not recovered, a global coverage of 99.948%. Of the 184
documents exposing text, 116 came out at exactly 100% and none below 99.5%. The
remaining four are scanned drawings containing no text at all in the file: they
were read by optical character recognition and are marked as unverifiable,
because no text original exists to measure them against.

**Packaging.** The corpus, its search index and the provenance of every passage
fit into a single `.mdcx` file, encrypted with AES-256-GCM, whose header can be
read without the key. From 8.8 MB of Markdown to 3.9 MB in one file.

**Retrieval.** A query returns the passages that answer it with their exact
source. Over the 20 real queries used for tuning, the correct document appears
within the top five results in 19 cases and within the top ten in all 20.

## Installation

The package separates querying from conversion, because they have very different
requirements.

| Command | Installs | Size |
|---|---|---|
| `pip install mdcx` | query and read `.mdcx` packages | ~10 MB |
| `pip install "mdcx[mcp]"` | the above plus the MCP server | ~50 MB |
| `pip install "mdcx[multilingual]"` | queries that cross languages | ~2.5 GB |
| `pip install "mdcx[convert]"` | document conversion (Docling, PyTorch) | ~1.4 GB |
| `pip install "mdcx[all]"` | everything, including OCR | ~4 GB |

Conversion is what pulls in the heavy dependencies. Someone who receives an
`.mdcx` file and only needs to query it installs neither Docling nor PyTorch.

The multilingual extra is what makes a query in one language reach documents
written in another, and most of its size is the model, downloaded once on first
use. A corpus written in a single language does not need it.

## Converting a collection

```
pip install "mdcx[convert]"
mdcx-convert --input ./Documents --output ./Documents_md
```

The output mirrors the input directory structure, adds a global index, and
records for each file the coverage achieved against its original.

### Formats

PDF, EPUB, Word, Excel, PowerPoint, HTML, Markdown, CSV and plain text.

Which one a file is gets decided by reading its first bytes rather than its
name. A file states what it is in its own content, and that statement is worth
more than its extension: repositories serve EPUB files from URLs ending in
`.pdf`, declaring `application/pdf`, where only the bytes disagree. Handed to
the PDF reader such a file fails in a way that looks like a damaged document
rather than a misrouted one, and it converts without its fidelity ever being
measured.

Plain text carries no signature, so there the extension decides, which is the
one case where the name holds information the content does not. A file whose
bytes identify nothing is left alone rather than guessed at.

## Packaging and querying

```
mdcx pack --output ./Documents_md --target corpus.mdcx --key "..."
mdcx info corpus.mdcx
mdcx search corpus.mdcx "where is the minimum diameter stated" --key "..."
mdcx export corpus.mdcx --target ./restored --key "..."
```

`info` reads the header without the key, so the issuer and the integrity of a
file can be checked before opening it. `export` rebuilds the original folder: a
format that cannot be left is a trap, however well intended.

## Using it as an MCP server

The server requires Python and this package. It does not require the conversion
stack, so the footprint is about 50 MB.

```json
{
  "mcpServers": {
    "mdcx": {
      "command": "python",
      "args": ["-m", "mdcx.mcp_server"],
      "env": {
        "MDCX_FILE": "/path/to/corpus.mdcx",
        "MDCX_KEY": "package-key"
      }
    }
  }
}
```

Alternatively, with [uv](https://docs.astral.sh/uv/) the server runs without a
prior installation, which is the usual arrangement for Python MCP servers:

```json
{
  "mcpServers": {
    "mdcx": {
      "command": "uvx",
      "args": ["--from", "mdcx[mcp]", "python", "-m", "mdcx.mcp_server"],
      "env": {
        "MDCX_FILE": "/path/to/corpus.mdcx",
        "MDCX_KEY": "package-key"
      }
    }
  }
}
```

Three tools are exposed. `search` returns the passages answering a question, each
with its source document and portable path. `info` describes the corpus and the
fidelity of its conversion. `document` returns a full document when passages are
not enough.

The server verifies the package before it starts listening, so a wrong path or
key is reported immediately rather than on the first query.

## Tests

```
pip install pytest
python -m pytest tests/ -v
```

Ninety-three tests run in four groups.

`test_stress.py` covers hostile inputs: empty and corrupted files, names in
other alphabets, malformed queries including SQL injection attempts, truncated
and tampered packages, concurrent access, and compaction against content loss.

`test_languages.py` measures retrieval itself. It builds a corpus of 136
documents written in 34 languages across 11 writing systems, and checks two
properties: that a query written in a language retrieves the documents written
in that language, and that a term shared by several languages returns the
documents of all of them. The second is what keeps a search from narrowing to
one language, whether the language of the query or that of the corpus.

`test_formats.py` checks that a file is treated as what it contains rather than
as what it is called, in both directions: a supported format under an unexpected
extension, and a file served under the wrong one.

`test_multilingual.py` measures retrieval across languages: that a query written
in one language returns documents written in others, that every writing system
is reachable, and that merging the two engines does not cost the precision the
lexical engine has on its own. These are skipped when the multilingual extra is
absent, which is a supported configuration rather than a failure.

## Languages

Retrieval is lexical and script-aware. A query matches the words that appear in
the documents, in whatever writing system they were written, and the results of
one search may come from documents in several different languages at once. The
language of a corpus is recorded and reported by `info`; it describes the corpus
and never restricts what a query returns.

The following are verified by `tests/test_languages.py`, which builds one corpus
holding the same four subjects — algebra, botany, printing and baking — written
in every language listed, and then issues a query in each language. Every query
competes against the three other documents in its own language and against the
whole of the rest of the corpus. All 136 queries return the expected document in
first place.

| Script | Languages |
|---|---|
| Latin | English, Spanish, Portuguese, French, Italian, German, Dutch, Swedish, Danish, Norwegian, Finnish, Polish, Czech, Hungarian, Romanian, Turkish, Indonesian, Vietnamese, Catalan |
| Cyrillic | Russian, Ukrainian, Bulgarian, Serbian |
| Greek | Greek |
| Arabic | Arabic, Persian |
| Hebrew | Hebrew |
| Devanagari | Hindi |
| Bengali | Bengali |
| Tamil | Tamil |
| Thai | Thai |
| Han | Chinese, Japanese |
| Hangul | Korean |

Support is a property of the writing system rather than of the language, so a
language written in one of these scripts is covered whether or not it appears in
the table. Three properties make that hold:

Words are runs of letters, digits and the combining marks that belong to them.
The marks are read from the Unicode database rather than listed, which is what
keeps the vowels of Devanagari, Bengali, Tamil and Thai attached to the letters
they modify instead of splitting each word into fragments.

Accent folding is limited to the combining marks that represent an accent placed
on a letter, so that `café` matches `cafe`. The vowel signs of Indic scripts and
the points of Hebrew and Arabic are left in place, because there they carry the
sound of the syllable rather than decorate it.

Writing systems that do not separate words with spaces — Chinese, Japanese,
Korean, Thai, Lao, Khmer, Burmese, Tibetan and Javanese — are indexed by
character, at query time and at index time alike. This is what a lexical index
can match without a segmenter trained on one particular language, which would
serve that language and leave every other one where it started.

Everything above concerns matching by word, which is matching within a language:
the words of a query have to be present in the document. A query written in one
language reaches a document written in another only where the two share a term,
as proper names and loanwords often do. When a query returns nothing and none of
its terms appear in the index, the result says so and names the language of the
corpus, so an empty answer can be told apart from material the corpus does not
hold.

Crossing between languages is a separate capability, described in the next
section. It is optional because it needs a model, and it is merged with word
matching rather than replacing it.

## Searching across languages

Word matching is matching within a language. A Spanish query and a German
document about the same subject share no word, so there is nothing for an index
of words to find, however well the words are tokenised. Measured on a corpus
written in thirty-four languages, a query in one language retrieves 4.2 per cent
of the documents on its subject: essentially only the ones written in the
language of the query.

Reaching the rest requires representing meaning rather than spelling. A
multilingual embedding model places a sentence and its translation near each
other, so a document can be found through what it says instead of the words it
happens to use. Built with `--multilingual`, a package stores a vector for each
passage next to the passage itself, under the same encryption, and the same
query then retrieves 96.9 per cent of them.

```
pip install "mdcx[multilingual]"
python -m mdcx.archive pack --output ./corpus_md --target corpus.mdcx \
    --key "..." --multilingual
```

The corpus is encoded once, when the package is built. Whoever receives it
encodes only their own queries.

### Both engines, merged

The two engines are kept side by side because they fail in opposite directions.
On the same corpus:

| engine | across languages | expected document ranked first |
|---|---|---|
| words | 4.2% | 136 of 136 |
| meaning | 98.5% | 126 of 136 |
| merged | 96.9% | 135 of 136 |

Meaning alone reaches almost everything and loses precision on the language of
the query, where an exact word is exactly what should decide. Words alone are
precise and cannot leave that language. Merged by reciprocal rank, each covers
what the other cannot, at the cost of one document out of a hundred and thirty
six against the lexical engine on its own ground.

Rank is merged rather than score because the two scales have no common meaning:
a BM25 score of 8 and a cosine similarity of 0.8 cannot be added, and
normalising them introduces a weighting that nothing justifies.

`--mode lexical` and `--mode semantic` select a single engine when the
comparison matters.

### Choosing the model

The default is `BAAI/bge-m3`, chosen by measurement on FLORES-200, a corpus of
sentences translated by professionals into two hundred languages. The task is
to find a sentence given its translation, among candidates from the same corpus,
in both directions of every language pair.

What decides is not the average but the pair of languages the model handles
worst. A model that averages well while collapsing on one language does not
serve whoever reads in that language.

| model | mean | worst language | worst pair |
|---|---|---|---|
| BAAI/bge-m3 | 100.0% | 99.8% | 98.0% |
| sentence-transformers/LaBSE | 99.5% | 97.7% | 96.0% |
| intfloat/multilingual-e5-large | 98.9% | 96.5% | 92.0% |
| intfloat/multilingual-e5-small | 97.0% | 94.4% | 92.0% |
| ibm-granite/granite-embedding-97m-multilingual-r2 | 95.4% | 91.3% | 84.0% |

Measured over 132 language directions covering ten writing systems, with 50
candidates per query. On a larger run of 1 122 directions across all
thirty-four languages with 100 candidates, LaBSE scored 99.6% mean and 93.5% on
its worst language, with no pair below 90%.

Another model can be used by name:

```
MDCX_MODEL=sentence-transformers/LaBSE python -m mdcx.archive pack ...
```

A package records the model that encoded it. A query encoded by a different
model lands elsewhere in the vector space, so the mismatch disables meaning
rather than returning results that look ranked and are not.

### What this does not do

Retrieval finds documents that say something close to the query. It does not
translate them: passages are returned in the language they were written in.
Nor does it make an unrelated document relevant because the model recognised the
subject; the ranking still has to place it, and the measurements above are what
it does place.

## Paths

No output contains absolute paths. Every document is identified by a pseudopath
beginning with `@/`, resolved against the folder or package containing it, so a
corpus remains valid wherever it is stored: local disk, network share or cloud.

## Signing

A package can be signed so that its issuer can be proven rather than merely
declared. The signature covers the digest of the encrypted body, so it attests
both origin and content, and is verified without the encryption key.

```
mdcx keygen
mdcx pack --output ./Documents_md --target corpus.mdcx --key "..." \
          --issuer "Acme Ltd" --signing-key <private-key>
mdcx verify corpus.mdcx --public-key <public-key>
```


Verification also requires the body to be intact: a signature covering only the
recorded digest would otherwise accept a package whose contents had been replaced
while its header was left untouched.

The issuer field alone is free text and proves nothing. Only a signature does.

## Encryption

The package encrypts at rest and decrypts in memory when opened; nothing is
written to disk in clear. This protects a file in transit. It is not the same as
searching over encrypted data without ever decrypting it, which is a separate
field with documented leakage attacks and per-query costs measured in seconds.

The key is derived with scrypt, which makes guessing slow: about 8 attempts per
second, each requiring 32 MB of memory, which prevents parallelisation on a GPU.
Even so, **the real strength is the passphrase**: a dictionary password falls in
a day.

## Authorship

Conceived and directed by **Jorge Ellena G.**, programmed with the assistance of
Claude (Anthropic).

Every decision in this package was made against measurements rather than
convention: which conversion engine to use, which licence permits which, how to
rank a search, which optimisations to accept and which to discard. Several were
discarded precisely because they were measured — reducing the search candidate
pool appeared to be ten times faster and in fact lowered accuracy from 19 to 17
out of 20 — and those measurements are recorded alongside the decisions they
justify.

## Citation

Archived on Zenodo with a permanent identifier. The concept DOI always resolves
to the latest version:

    https://doi.org/10.5281/zenodo.22015991

## Licence

Apache 2.0. The software may be used, modified and sold, provided the copyright
notice is retained.

PyMuPDF was deliberately avoided: its AGPL licence would require anyone using
this software to publish their own under AGPL, including those offering it only
as a network service.
