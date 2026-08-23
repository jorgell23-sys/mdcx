# mdcx

<!-- mcp-name: io.github.jorgell23-sys/markdown-document-search -->

[![PyPI](https://img.shields.io/pypi/v/mdcx)](https://pypi.org/project/mdcx/) [![tests](https://github.com/jorgell23-sys/mdcx/actions/workflows/tests.yml/badge.svg)](https://github.com/jorgell23-sys/mdcx/actions/workflows/tests.yml) [![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/jorgell23-sys/mdcx/blob/main/LICENSE) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22015991.svg)](https://doi.org/10.5281/zenodo.22015991)

Convert a document collection to verified Markdown, package it into a single
encrypted file, and query it from an agent through the Model Context Protocol.

## Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Conversion](#conversion)
- [Packaging and querying](#packaging-and-querying)
- [Sent and received](#sent-and-received)
- [Working incrementally](#working-incrementally)
- [MCP server](#mcp-server)
- [Language support](#language-support)
- [Cross-language retrieval](#cross-language-retrieval)
- [Portable paths](#portable-paths)
- [Signing](#signing)
- [Encryption](#encryption)
- [Limitations](#limitations)
- [Tests](#tests)
- [Contributing](#contributing)
- [Security](#security)
- [Releases](#releases)
- [Authorship](#authorship)
- [Citation](#citation)
- [Licence](#licence)

## Overview

mdcx converts a collection of documents to Markdown, verifies each conversion
against its original, packages the corpus with its index and provenance into a
single encrypted file, and serves that file to agents over the Model Context
Protocol.

It addresses one constraint. An agent asked a question about a document
collection must either receive the documents in its context window, which is
bounded in size and billed per token, or query a component that holds an index
and returns only the passages that bear on the question. mdcx implements the
second. Three properties distinguish it from an extraction script:

- **Fidelity is measured, not assumed.** Every conversion is checked against the
  text the original exposes, read by a library independent of the engine that
  produced the conversion, and the coverage achieved is recorded per file.
- **The corpus is a single encrypted artefact.** Passages, index and provenance
  are held in one AES-256-GCM file whose header can be read without the key.
- **Every passage carries its source.** An answer can be cited against a
  document and a location rather than recalled.

### Pipeline

**Conversion.** Each document is attempted by the least expensive engine capable
of reading it and escalated only where that engine falls short: direct text
extraction, then a pass that recovers the tables a page draws, then full layout
analysis. Documents exposing no text are read by optical character recognition.
Content the selected engine omitted is appended verbatim rather than reported as
lost.

Over the collection used during development — 99 documents, 1,144,553 reference
tokens — 594 tokens were not recovered, a coverage of 99.948%. Of the 95
documents that expose text, 70 were recovered in full and none fell below 99.5%.
The remaining four are scanned drawings holding no text in the file; they are
marked unverifiable, as no text original exists to measure them against.

**Packaging.** The corpus, its search index and the provenance of every passage
are written to a single `.mdcx` file. The development collection produced 3.9 MB
from 8.8 MB of Markdown. A growing collection is not rebuilt from the start:
vectors already computed are reused, and a corpus exceeding what can be
decrypted into memory is held as several packages queried as one.

**Retrieval.** A query returns the passages that answer it, each with its source
document and its position in the ranking. Word matching and dense retrieval are
merged by reciprocal rank, so a query reaches a document whether it shares that
document's vocabulary or only its subject, including where the two are written
in different languages. Over a corpus of 136 documents in 34 languages, the
merged engines rank the expected document first for 135 of the 136 queries.
Where no document in the corpus is about the question, the reply states this
rather than presenting its nearest passage as an answer.

### Measured cost

One query over the development collection — 99 documents, 180 MB — counted with
the `cl100k_base` tokenizer:

| Method | Model tokens | Local tokens |
|---|---|---|
| Reading the originals | 2,265,488 | 2,265,327 |
| Querying the package | 435 | 2,688,861 |

The 435 model tokens comprise 20 for the question, 274 for the retrieved passage
and 141 for the answer.

Reading the originals costs the whole collection because a PDF is a binary
format: absent prior conversion there is no way to determine which of the 99
documents holds the answer, so all of them are extracted and read.

This is a single measurement, not an average, and the saving depends on how much
text an answer requires. The work is not eliminated but relocated, from the
context window, which is billed and finite, to local processing, which is
neither. The local column rises for that reason.

## Requirements

Python 3.11 or later. No other component is required to query a package.

The floor is 3.11 because a package is held as one SQLite database and
serialised in memory to be encrypted, and `sqlite3` gained the call that
does so in that version. Earlier interpreters were declared supported and
were not: neither building a package nor opening one worked there.
Conversion and cross-language retrieval each add dependencies, listed under
[Installation](#installation).

## Installation

Querying and conversion are separated because their requirements differ by two
orders of magnitude.

| Command | Provides | Approximate size |
|---|---|---|
| `pip install mdcx` | querying and reading `.mdcx` packages | 10 MB |
| `pip install "mdcx[mcp]"` | the above and the MCP server | 50 MB |
| `pip install "mdcx[convert]"` | document conversion (Docling, PyTorch) | 1.4 GB |
| `pip install "mdcx[tables]"` | tables a page does not draw | 1.2 GB |
| `pip install "mdcx[multilingual]"` | cross-language retrieval | 2.5 GB |
| `pip install "mdcx[all]"` | all of the above, including OCR | 4 GB |

Conversion accounts for the heavy dependencies. A recipient who only queries an
`.mdcx` file installs neither Docling nor PyTorch.

The `multilingual` extra is required for queries that cross languages. Most of
its size is the embedding model, downloaded once on first use. A single-language
corpus does not require it.

The `tables` extra covers what a page does not draw. Tables in printed material
are usually found from the rules drawn around them, which costs nothing and
needs no extra; borderless ones — a screenshot of a spreadsheet, a layout held
together by alignment — are read by a small model that reports where the rows
and columns run. It reads the shape only: the words still come from the text
layer of the document, so a cell cannot hold anything the page does not say.
Without it those pages are read by Docling instead, which is slower but already
present in the `convert` extra.

## Quick start

```
pip install "mdcx[convert]"

mdcx-convert --input ./Documents --output ./Documents_md
mdcx pack --output ./Documents_md --target corpus.mdcx --key "passphrase"
mdcx search corpus.mdcx "where is the storage temperature stated" --key "passphrase"
```

## Conversion

```
mdcx-convert --input ./Documents --output ./Documents_md
```

The output mirrors the input directory structure, adds a global index, and
records for each file the coverage achieved against its original.

### Supported formats

PDF, EPUB, Word, Excel, PowerPoint, HTML, Markdown, CSV and plain text.

The format of a file is determined from its first bytes rather than from its
extension. Repositories are known to serve EPUB files from URLs ending in `.pdf`
and declaring `application/pdf`, where only the content identifies the format
correctly. Routing such a file by extension sends it to a reader that cannot open
it, and the resulting failure is indistinguishable from a damaged document.

Plain text carries no signature, so its extension determines the format. A file
whose content identifies no known format is skipped rather than assumed.

### How much of the machine it uses

Converting a library is the heaviest thing this package does, and it runs on a
machine somebody is working at. A fifth of the processors are left free and the
rest are used, so the figure follows the machine rather than a constant: nine
processes on twelve processors, three on four.

```
mdcx-convert --input ./Documents --output ./Documents_md --max-cores 4
```

The cap is a budget for the whole run rather than a grant to each process. It is
divided among the workers and each is told its share, because the structured
engine asks for four threads of its own: eight workers taking that at face value
would ask for thirty-two threads on a machine of twelve, and the share that was
carefully left free would be spent again inside the pool.

Documents are dispatched to two lanes. The GPU lane is small on purpose and is
the only one allowed to use the card, because video memory is what runs out
first: twelve processes each loading the table models reached 5,893 MiB of 6,144
and ran three times slower than three processes did. It receives the documents
that expose no text, which have nothing to extract and must be read by optical
recognition.

Everything else goes to the larger lane and runs on the processor, with the same
engines available to it. The lane decides which device the work runs on, not
which engines exist: a document that needs layout analysis gets it in either
lane, on the card in one and on the processor in the other.

### Checking a conversion before packaging

`mdcx-search` searches the converted Markdown directly, before there is a
package, and quotes each passage with the document and pseudopath it came from.
It is how a conversion is inspected while the folder is still open to
correction.

```
mdcx-search "movable type" --output ./Documents_md
mdcx-search --phrases ./questions.txt --output ./Documents_md --json found.json
```

Passages are ranked with BM25 aggregated per document rather than in isolation,
so a long document that covers a subject across several fragments is not beaten
by a short unrelated one that repeats a term. `--literal` requires the exact
phrase and nothing else; `--bm25` ranks by relevance without literal matching.

This engine reads the Markdown folder. Retrieval over a built package, with
meaning and across languages, is `mdcx search` and the MCP server.

## Packaging and querying

```
mdcx pack --output ./Documents_md --target corpus.mdcx --key "passphrase"
mdcx info corpus.mdcx
mdcx search corpus.mdcx "where is the storage temperature stated" --key "passphrase"
mdcx export corpus.mdcx --target ./restored --key "passphrase"
```

`info` reads the header without the key, so the issuer and the integrity of a
file can be checked before it is opened. `export` reconstructs the original
folder, so a collection can be moved out of the format at any time.

## Sent and received

Correspondence has a direction, and a question about it is usually about one
side: what was asked of us, or what we answered. Where the top-level folder of
a collection states that direction, it is recorded per document and a query can
be restricted to it.

| Top-level folder contains | Direction |
|---|---|
| `sent`, `emitido`, `outgoing` | sent |
| `received`, `recibido`, `incoming` | received |
| anything else | unclassified |

The names are recognised in English and Spanish, since a collection may be
organised in either, and only the top-level folder is examined, so a subfolder
named after a correspondent does not reclassify what it holds.

```
mdcx search corpus.mdcx "what was agreed about the schedule"     --key "passphrase" --only received
```

The MCP `search` tool takes the same restriction as its `direction` argument.
A collection organised in any other way is unaffected: every document is
unclassified, and a query that names no direction is not narrowed.

## Working incrementally

A collection that is delivered once and a collection that grows every day place
different demands on the tool. The second must not pay for what it has already
done.

### Conversion resumes

Conversion records the digest of each source in the Markdown it produces, and
skips any file whose source is unchanged and whose output is present. This is
the default; `--force` disables it.

The unit is the chapter rather than the document, so a book split into 47
chapters and interrupted at the 40th costs the remaining 7 on the next run.
Progress is written after each unit and flushed to disk, so an interrupted run
leaves a record that the next one reads.

A run over a converted collection reports what it reused:

```
Already converted and unchanged: 8 (reused)
Elapsed         : 0.4 min
```

A chapter is reconverted when its verification reported findings, since a result
that was not certified is not a result worth keeping.

### Packaging costs what was added

Indexing meaning dominates the cost of packaging. On one measured book: 505
seconds of encoding against 4 seconds of compression and 0.2 of encryption.
Encoding the whole corpus on every publication makes adding one document cost a
reindex of every previous one.

A passage whose text has not changed has the same vector. `--reuse` reads the
vectors of an existing package and encodes only what is new:

```
mdcx pack --output ./Documents_md --target corpus-2.mdcx --key "passphrase" \
    --multilingual --reuse corpus-1.mdcx
```

```
  meaning indexed with BAAI/bge-m3 (1024 dimensions)
  passages encoded 395   reused 733
```

Measured over the chapters of one book, where 733 of 1,128 passages were
unchanged, packaging took 15.4 seconds against 37.8 without reuse.

The vectors are read from the previous package, which already holds them and is
already encrypted with the same key. No intermediate store is created: a vector
allows the text it represents to be approximated, so keeping vectors outside the
package would undo the encryption the format provides.

Reuse requires the same model. Vectors from two models occupy different spaces,
so a package encoded by another model contributes nothing rather than
contributing values that cannot be compared.

### Several packages as one corpus

`MDCX_FILE` accepts more than one package, separated by the path separator of
the platform or by a comma. The server queries all of them and returns one
ranked list, with each result naming the package it came from.

```json
{
  "mcpServers": {
    "mdcx": {
      "command": "python",
      "args": ["-m", "mdcx.mcp_server"],
      "env": {
        "MDCX_FILE": "/corpora/2026-01.mdcx:/corpora/2026-02.mdcx",
        "MDCX_KEY": "package-key"
      }
    }
  }
}
```

One key serves every package; several keys are matched to the packages in order.

This makes each package immutable: it is indexed once and never rebuilt. A
corpus grows by adding packages rather than by enlarging one, which also keeps
each of them within what can be decrypted into memory, since a package is
decrypted whole when it is opened.

Results from different packages are merged by reciprocal rank. Their scores are
computed over different corpus statistics — the frequency of a term depends on
the corpus it is measured in — so the scores are not comparable between
packages, while positions within each are.

## MCP server

The server requires Python and this package. It does not require the conversion
stack; its footprint is approximately 50 MB.

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

With [uv](https://docs.astral.sh/uv/) the server runs without prior installation,
which is the common arrangement for Python MCP servers:

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

Three tools are exposed:

| Tool | Returns |
|---|---|
| `search` | passages answering a question, each with its source document, portable path and rank; `direction` restricts it to one side of a correspondence |
| `info` | the corpus record, including the fidelity of its conversion |
| `document` | a complete document, when passages are insufficient |

The package is verified before the server begins listening, so an incorrect path
or key is reported at startup rather than on the first query.

A passage carries its `rank` and no score. The list is ordered by that rank and
by nothing else: word matching and meaning score on scales with no common
meaning — one has no upper bound and depends on the corpus it was measured in,
the other runs from zero to one and does not — so there is no single number here
that can be compared, sorted or filtered by.

What can be compared is reported once for the reply. `similarity` is how near
the corpus comes to the question, and a `warning` appears when nothing in it is
about the question. The passages are returned either way: the nearest passage is
worth seeing even when it is not an answer, and a corpus that answers in another
language must not be hidden by this.

## Language support

Retrieval by word is script-aware. A query matches the words present in the
documents, in any writing system, and the results of a single search may include
documents in several languages. The predominant language of a corpus is recorded
and reported by `info`; it describes the corpus and does not restrict what a
query returns.

The following are verified by `tests/test_languages.py`, which builds one corpus
holding the same four subjects — algebra, botany, printing and baking — in every
language listed, then issues a query in each. Each query competes against the
three other documents in its own language and against the remainder of the
corpus. All 136 queries return the expected document in first position.

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
language written in any of these scripts is covered whether or not it is listed.
Three properties establish this:

**Tokenisation.** A word is a run of letters, digits and the combining marks
attached to them. The set of combining marks is derived from the Unicode
character database rather than enumerated, which keeps the vowel signs of
Devanagari, Bengali, Tamil and Thai attached to the letters they modify.

**Accent folding.** Folding is restricted to combining marks that represent an
accent placed on a letter, so that `café` matches `cafe`. The vowel signs of
Indic scripts and the points of Hebrew and Arabic are preserved, since in those
scripts they carry the sound of the syllable.

**Segmentation.** Writing systems that do not separate words with spaces —
Chinese, Japanese, Korean, Thai, Lao, Khmer, Burmese, Tibetan and Javanese — are
indexed by character, at index time and query time alike. This is what a lexical
index can match without a segmenter trained on a single language.

Word matching operates within a language: the words of a query must be present in
the document. A query written in one language reaches a document written in
another only where the two share a term, as proper names and loanwords often do.
When a query returns no result and none of its terms appear in the index, the
response states this and names the language of the corpus, distinguishing an
empty answer from material the corpus does not hold.

Retrieval across languages is a separate capability, described below. It is
optional and requires a model. Where the query is written in the language of the
documents the two are merged, each covering what the other cannot; where it is
not, word matching has nothing to contribute and is left out, because the few
terms it does match there are accidents and they arrive first.

## Cross-language retrieval

Word matching operates within a language. A Spanish query and a German document
on the same subject share no term, so a word index has nothing to match. Measured
on a corpus written in 34 languages, a query retrieves 4.2% of the documents on
its subject, comprising essentially those written in the language of the query.

Retrieving the remainder requires representing meaning rather than spelling. A
multilingual embedding model places a sentence and its translation at nearby
points in a vector space, so a document can be retrieved through its content
rather than its vocabulary. When built with `--multilingual`, a package stores a
vector for each passage alongside the passage, under the same encryption, and the
same query retrieves 96.9% of the documents.

```
pip install "mdcx[multilingual]"
mdcx pack --output ./Documents_md --target corpus.mdcx \
    --key "passphrase" --multilingual
```

The corpus is encoded once, when the package is built. A recipient encodes only
their own queries.

### Merged engines

Both engines are retained because their failure modes are complementary. Measured
on the same corpus of 136 documents in 34 languages:

| Engine | Across languages | Expected document ranked first |
|---|---|---|
| Word | 4.2% | 136 of 136 |
| Meaning | 98.5% | 126 of 136 |
| Merged | 96.9% | 135 of 136 |

The dense engine retrieves across languages but ranks less precisely within the
language of the query. The lexical engine ranks precisely and does not retrieve
beyond that language. Merging by reciprocal rank retains both properties, at a
cost of one document in 136 relative to the lexical engine alone.

Ranks are merged rather than scores, as a BM25 score and a cosine similarity have
no common scale.

`--mode lexical` and `--mode semantic` select a single engine.

### Model selection

The default is `BAAI/bge-m3`. Models were compared on FLORES-200, a corpus of
sentences translated by professionals into 200 languages. The task is to retrieve
a sentence given its translation, among candidates drawn from the same corpus, in
both directions of every language pair.

The selection criterion is the worst-performing language pair rather than the
mean, since a mean can conceal a language on which a model performs poorly.

| Model | Mean | Worst language | Worst pair |
|---|---|---|---|
| `BAAI/bge-m3` | 100.0% | 99.8% | 98.0% |
| `sentence-transformers/LaBSE` | 99.5% | 97.7% | 96.0% |
| `intfloat/multilingual-e5-large` | 98.9% | 96.5% | 92.0% |
| `intfloat/multilingual-e5-small` | 97.0% | 94.4% | 92.0% |
| `ibm-granite/granite-embedding-97m-multilingual-r2` | 95.4% | 91.3% | 84.0% |

Measured over 132 language directions covering 10 writing systems, with 50
candidates per query. In a larger run of 1,122 directions across all 34 languages
with 100 candidates, LaBSE reached a mean of 99.6% and 93.5% on its worst
language, with no pair below 90%.

An alternative model is selected by name:

```
MDCX_MODEL=sentence-transformers/LaBSE mdcx pack ...
```

A package records the model that encoded it. A query encoded with a different
model occupies a different vector space, so a mismatch disables meaning-based
retrieval rather than returning results that cannot be compared.

## Portable paths

No output contains absolute paths. Each document is identified by a pseudopath
beginning with `@/`, resolved against the folder or package containing it, so a
corpus remains valid on local disk, network share or cloud storage.

## Signing

A package can be signed so that its issuer can be verified rather than declared.
The signature covers the digest of the encrypted body, attesting to both origin
and content, and is verified without the encryption key.

```
mdcx keygen
mdcx pack --output ./Documents_md --target corpus.mdcx --key "passphrase" \
          --issuer "Acme Ltd" --signing-key <private-key>
mdcx verify corpus.mdcx --public-key <public-key>
```

Verification requires the body to be intact. A signature covering only the
recorded digest would accept a package whose contents had been replaced while its
header was left unmodified.

The issuer field is free text and is not evidence of origin on its own.

## Encryption

Packages are encrypted at rest and decrypted in memory when opened; no plaintext
is written to disk. This protects a file in transit and at rest. It is not
equivalent to searching over encrypted data without decryption, which is a
distinct field with documented leakage attacks and per-query costs measured in
seconds.

The key is derived with scrypt at N = 2^15, r = 8, p = 1. Those parameters
require 32 MB of memory per attempt, which is what resists the parallelisation a
GPU would otherwise bring to a search: memory, unlike arithmetic, does not
become cheap by adding cores. One derivation takes 286 ms single-threaded on the
development machine, approximately 3.5 attempts per second per core.

That cost falls on an attacker and on the legitimate opening of a package alike.
It multiplies the work of a search; it does not make a weak passphrase safe. The
strength of the encryption is the strength of the passphrase, and one drawn from
a dictionary stays within reach of an offline search whatever the derivation
costs.

## Limitations

Retrieval returns documents whose content is close to the query. It does not
translate them: passages are returned in the language in which they were written.

Documents that expose no text, such as scanned drawings, are read by optical
character recognition and counted as unverifiable rather than as findings, since
no text original exists against which to measure fidelity. Coverage is computed
over the documents that could be measured, so an unverifiable document neither
raises nor lowers it.

Coverage measures the tokens preserved by a conversion. It does not measure the
preservation of table structure, which is reported separately.

Deciding that a grid of drawn rules is a table, rather than prose someone framed
for emphasis, is done by how many of its rows run to more than one line. That
test is not exact in either direction: prose laid out in short lines can pass it,
and a table whose cells wrap can fail it. A separating signal was looked for in
the material at hand and not found — the longest cell in the sample belongs to a
legitimate table, so cell length does not divide them — and no further threshold
was added on the strength of one collection. Where the decision goes wrong, the
text is still present and still counted in coverage; what is lost is its shape.

A package is decrypted in full when it is opened, so its size is bounded by the
memory available. A corpus larger than that is held as several packages and
queried together, as described under
[Working incrementally](#working-incrementally).

## Tests

```
pip install pytest
python -m pytest tests/ -v
```

254 tests. Most of them exist because something failed once; the file that
covers it says which, so a correction that is undone is noticed.

**Retrieval**

| File | Scope |
|---|---|
| `test_languages.py` | retrieval in 34 languages across 11 writing systems, and the requirement that a term shared by several languages returns the documents of all of them |
| `test_multilingual.py` | retrieval across languages, and the requirement that merging engines preserves the precision of the lexical engine |
| `test_multipackage.py` | querying several packages as one corpus, including key configuration and the reporting of a missing package |
| `test_relevance.py` | that serving several packages does not bury the answer among the ones that hold nothing about it |
| `test_direction.py` | restricting a search to one side of a correspondence, and the requirement that both engines filter by the same form of the value |
| `test_answer_quality.py` | what the server says when it cannot answer, and the places where a number meant something other than it appeared to |
| `test_describes_itself.py` | that every field a reply carries and every argument a tool accepts are accounted for in the description the caller reads, and that no field is promised after it stopped arriving |
| `test_encoder.py` | how the encoder spends the accelerator, and that batching leaves the vectors unchanged |
| `test_package_identity.py` | that a cache belongs to a package rather than to a memory address, so a package cannot answer with the vectors or the corpus statistics of one that was closed |

**Conversion**

| File | Scope |
|---|---|
| `test_formats.py` | identification of a file by content rather than extension, in both directions, and the extraction of reference text from EPUB |
| `test_conversion_order.py` | which engine reads a document, when the search for a better one stops, and what rejects a table that is not one |
| `test_single_extraction.py` | that a document is extracted once however many engines read it, and that what each is given is its own to edit |
| `test_table_cells.py` | that a character of a drawn table lands in exactly one cell, including a glyph the outer rule cuts |
| `test_table_shapes.py` | the reading of a table the page does not draw, where the model supplies the shape and the text layer the words |
| `test_headings.py` | that a chapter keeps the section titles its book already carried, whichever engine converted it |
| `test_reporting.py` | that the summary separates a document measured and found short from one that could not be measured at all |
| `test_incremental.py` | reuse of vectors between packages: that unchanged passages are not encoded again, that an edited one is, and that reuse produces the same ranking |

**The package as it is installed**

| File | Scope |
|---|---|
| `test_stress.py` | hostile inputs: empty and corrupted files, names in other alphabets, malformed queries including SQL injection, truncated and tampered packages, concurrent access, and compaction against content loss |
| `test_entrypoints.py` | that every command the package declares can be started and can report its version, and that every MCP tool publishes the signature of the function that answers it |
| `test_machine_share.py` | that a conversion leaves a share of the machine free, counting threads as well as processes; that the lanes are sized from the video memory, the work and the processors rather than from constants; and that a turn on the card is given back even when the model raises |
| `test_console.py` | that a document name the console cannot represent does not stop the conversion, in the parent process and in the workers |

Tests that need a model skip themselves when it is absent, so the suite passes
on a plain `pip install mdcx` as well as on `[all]`. Seven files depend on the
`multilingual` extra and one on `tables`.

Continuous integration runs the suite on Python 3.11 and 3.13, on Linux and
Windows.

## Contributing

Issues and pull requests are accepted at
[github.com/jorgell23-sys/mdcx](https://github.com/jorgell23-sys/mdcx).

A change needs a test that has been seen to fail without it, and a change about
cost or quality needs the measurement that justifies it. [CONTRIBUTING.md](CONTRIBUTING.md)
sets out what makes a report act on itself and what a pull request is expected
to carry; [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) covers the rest.

## Security

To report a vulnerability, open a security advisory at
[github.com/jorgell23-sys/mdcx/security/advisories](https://github.com/jorgell23-sys/mdcx/security/advisories)
rather than a public issue.

Packages are encrypted with AES-256-GCM and keys derived with scrypt. The
encryption protects a package at rest and in transit; it does not protect against
a compromised host, where the key is present in memory while the package is open.

## Releases

Version history and release notes:
[github.com/jorgell23-sys/mdcx/releases](https://github.com/jorgell23-sys/mdcx/releases).

Versioning follows [Semantic Versioning](https://semver.org/). The `.mdcx` format
is read backwards-compatibly: a package written by an earlier version remains
readable by a later one.

## Authorship

Conceived and directed by Jorge Ellena G., implemented with the assistance of
Claude (Anthropic).

Design decisions in this package are recorded alongside the measurements that
justify them, including the ones that were rejected. Several constants here are
the third value that was tried, and the two that failed are written down beside
them so the next attempt starts somewhere new. Where no measurement separated
two options, that is recorded too, rather than settled by a heuristic that
happened to fit the material at hand.

## Citation

Archived on Zenodo with a permanent identifier. The concept DOI resolves to the
latest version:

    https://doi.org/10.5281/zenodo.22015991

## Licence

Apache 2.0. See
[LICENSE](https://github.com/jorgell23-sys/mdcx/blob/main/LICENSE). Third-party
components and their licences are listed in
[NOTICE](https://github.com/jorgell23-sys/mdcx/blob/main/NOTICE).

PyMuPDF is not used. Its AGPL licence would require software incorporating this
package to be published under AGPL, including software offered as a network
service.
