# mdcx

<!-- mcp-name: io.github.jorgell23-sys/markdown-document-search -->

[![PyPI](https://img.shields.io/pypi/v/mdcx)](https://pypi.org/project/mdcx/) [![Python](https://img.shields.io/pypi/pyversions/mdcx)](https://pypi.org/project/mdcx/) [![tests](https://github.com/jorgell23-sys/mdcx/actions/workflows/tests.yml/badge.svg)](https://github.com/jorgell23-sys/mdcx/actions/workflows/tests.yml) [![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22015991.svg)](https://doi.org/10.5281/zenodo.22015991)

Convert a document collection to verified Markdown, package it into a single
encrypted archive with its search index and provenance, and serve it to agents
over the Model Context Protocol.

## Contents

- [Scope](#scope)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Conversion](#conversion)
- [Packaging](#packaging)
- [Querying](#querying)
- [Retrieval model](#retrieval-model)
- [Calibration](#calibration)
- [Vocabulary](#vocabulary)
- [Transcription recovery](#transcription-recovery)
- [Expressions](#expressions)
- [Quotations across a cut](#quotations-across-a-cut)
- [Attachments](#attachments)
- [Dates](#dates)
- [Correspondence](#correspondence)
- [Incremental packaging](#incremental-packaging)
- [Resident conversion](#resident-conversion)
- [MCP server](#mcp-server)
- [Sources](#sources)
- [Portable paths](#portable-paths)
- [Signing](#signing)
- [Encryption](#encryption)
- [Package format](#package-format)
- [Limitations](#limitations)
- [Tests](#tests)
- [Contributing](#contributing)
- [Security](#security)
- [Releases](#releases)
- [Authorship](#authorship)
- [Citation](#citation)
- [Licence](#licence)

## Scope

mdcx addresses one constraint. An agent asked a question about a document
collection must either receive the documents in its context window, which is
bounded and billed per token, or query a component that holds an index and
returns only the passages that bear on the question. mdcx implements the second.

Three properties define the result:

- **Fidelity is checked.** Each conversion is compared against the text the
  original exposes and the coverage achieved is recorded per file. Files that
  expose no text are marked unverifiable rather than reported as complete. What
  the comparison establishes differs by engine, and
  [Verification](#verification) states which.
- **The corpus is one artefact.** Passages, index and provenance are held in a
  single AES-256-GCM file whose header can be read without the key, and which
  may be signed.
- **Answers are citable.** Every passage carries its source document and its
  position, so an answer can be quoted against a location.

The pipeline has three stages. Conversion attempts each document with the least
expensive engine able to read it and escalates only where that engine falls
short: direct text extraction, then a pass that recovers drawn tables, then
layout analysis, and optical character recognition for documents that expose no
text. Packaging writes the corpus, its index and the provenance of each passage
to one file. Retrieval merges word matching and dense retrieval by reciprocal
rank, and states when nothing in the corpus bears on the question rather than
returning its nearest passage.

## Requirements

Python 3.11 or later. No other component is required to query a package.

The floor is 3.11 because a package is one SQLite database, loaded from the
decrypted bytes when it is opened, and `sqlite3` gained the required call in that
version. Conversion and cross-language retrieval each add dependencies, listed
under [Installation](#installation).

## Installation

Querying and conversion are separated because their requirements differ by
roughly two orders of magnitude. A recipient who only reads packages installs
neither Docling nor PyTorch.

| Command | Provides | Approximate size |
|---|---|---|
| `pip install mdcx` | querying and reading `.mdcx` packages | 10 MB |
| `pip install "mdcx[mcp]"` | the above and the MCP server | 50 MB |
| `pip install "mdcx[convert]"` | document conversion | 1.4 GB |
| `pip install "mdcx[tables]"` | tables a page does not draw | 1.2 GB |
| `pip install "mdcx[multilingual]"` | cross-language retrieval | 2.5 GB |
| `pip install "mdcx[all]"` | all of the above, including OCR | 4 GB |
| `pip install "mdcx[all-gpu]"` | the above without the CPU `onnxruntime` | 4 GB |

The `multilingual` extra supplies the embedding model, downloaded once on first
use, and is required only for queries that cross languages.

The `tables` extra covers tables that are not drawn with rules — a screenshot of
a spreadsheet, or a layout held together by alignment. It reads the geometry
only; cell contents still come from the text layer of the document. Without it
those pages are handled by the layout engine.

### Systems with a CUDA device

Install `mdcx[all-gpu]` rather than `mdcx[all]`, and install `onnxruntime-gpu`
separately.

`onnxruntime` and `onnxruntime-gpu` publish the same module and cannot coexist:
whichever was installed last takes effect. An extra that pins the CPU build will
therefore displace an accelerated one on every upgrade. `all-gpu` is `all`
without that pin.

pip cannot express a dependency satisfied by either distribution, so this cannot
be settled by declaration alone. `mdcx-convert` checks at startup and reports
when the machine has a device the runtime does not offer. The check on its own:

```
python -c "import onnxruntime as o; assert 'CUDAExecutionProvider' in o.get_available_providers()"
```

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

The output folder mirrors the input directory structure, one `.md` per source
document, together with an index of the run.

| Option | Effect |
|---|---|
| `--max-cores N` | processes to run at once |
| `--gpu-workers N`, `--cpu-workers N` | override the computed split |
| `--serial` | one document at a time |
| `--only PATTERN`, `--limit N` | restrict the run |
| `--force` | ignore the cache and reconvert |
| `--no-docling` | native engines only |
| `--no-gpu` | do not use the device |
| `--no-lossless` | do not write the backup JSON |
| `--no-compact` | keep converter scaffolding in the Markdown |
| `--no-split`, `--split-threshold N` | control splitting of long documents |
| `--sample-pages N` | convert a spread sample of N pages per document |
| `--timeout SECONDS` | abandon a document that exceeds this |
| `--recycle-after N` | replace a worker after N documents |

An interrupted run resumes: work already recorded is not repeated unless
`--force` is given.

Long documents are split into chapters, each converted independently, with an
index document recording the correspondence to the original. A chapter is
converted from an extract of the document, and its page markers state the
document's own numbering rather than the extract's; its front matter declares
`first_page` and `last_page`, so a passage can be cited back to the page it came
from. The index document declares `type: document_index` and holds no text of
its own — the text is in the chapters. `--sample-pages`
converts a spread sample instead of the whole document; the front matter records
both the pages taken and the document's total, so a sample is not mistaken for a
short document.

`--timeout` exists because some documents do not complete under layout analysis.
Without a limit one such document holds a worker for the length of the run.
`--recycle-after` replaces a worker periodically, since abandoning a document
does not stop the thread it started.

### Verification

Each conversion is compared against the text the original exposes. The index
records the coverage achieved per file. Documents that expose no text — scanned
drawings, for example — are marked unverifiable, since no text original exists to
measure against.

The reference is read by a library independent of the layout and OCR engines, so
for those the comparison is between two different readings of the document. For
native extraction it is not: the reference and the extraction are the same read,
and the comparison measures what the subsequent structuring and compaction kept.
The reference is read once and used for both.

## Packaging

```
mdcx pack --output ./Documents_md --target corpus.mdcx --key "passphrase"
```

| Option | Effect |
|---|---|
| `--key`, `--key-file` | the passphrase; see [Encryption](#encryption) |
| `--multilingual` | also index meaning |
| `--focus QUESTION` | a question the package exists to answer; repeatable |
| `--dates FILE` | supply publication dates; see [Dates](#dates) |
| `--date-from-mtime` | fall back to file modification time |
| `--shapes` | build the transcription-recovery index |
| `--expressions` | index the expressions the word rule discards |
| `--quotes document\|boundary` | how a quotation across a cut is found |
| `--reuse PACKAGE` | reuse vectors from an existing package |
| `--issuer`, `--signing-key` | see [Signing](#signing) |
| `--fast` | compress for speed rather than size |
| `--preset 0-9` | the compression level, overriding `--fast` |
| `--compression lzma\|zstd` | what compresses the body |

`--output` accepts a folder of Markdown, a single file, or a `.jsonl` file with
one record per line, which is how a catalogue of records is packed without first
writing it to disk.

`--fast` selects a lower compression preset, and `--preset` names one directly.
Sealing is a fixed price paid on every write: a package meant to be distributed
is compressed once and downloaded many times, while a corpus rebuilt whenever it
grows pays the clock and not the bytes. The level travels inside the compressed
stream, so a package written at any level is opened by any reader, and nothing
else about the package changes.

`--compression` chooses what compresses the body. The header records the name
and a package is read by what its header names, so existing packages are
unaffected and either kind is opened by a reader that has the module for it.
LZMA is the default and needs nothing; zstd needs `mdcx[zstd]`. Which is
smaller depends on the material — measured on one corpus zstd was smaller, on
another larger — while zstd is consistently faster to read back, which is what
a server pays before it can answer anything.

`pack` returns `seconds_index` and `seconds_seal`, and `seconds_index_by_phase`
divides the first among reading the folder, cutting passages, counting terms,
building the shape index and encoding. With `--multilingual` the encoding
dominates the rest by two orders of magnitude, and what it costs is set by how
long the passages are rather than how many: the model is charged per token.

The figure to plan a large corpus with is therefore a rate *and* the passage
length it was measured at. On one 6 GB card, encoding 600 passages of each
length:

| words per passage | passages/s |
|---:|---:|
| 10 | 431 |
| 20 | 196 |
| 50 | 84 |
| 90 | 52 |
| 200 | 21 |

A rate quoted without its passage length is not usable, and measuring one
against text generated for the purpose gives the machine rather than the work:
a consumer comparing 440 passages/s from invented 130-character texts against 25
from their own 942-character passages read the difference as a defect in this
library. The phase figures are what settle it for a given corpus.

## Querying

```
mdcx search corpus.mdcx "the question" --key "passphrase"
mdcx info corpus.mdcx
mdcx export corpus.mdcx --target ./restored --key "passphrase"
```

| Option | Effect |
|---|---|
| `--limit N` | passages to return |
| `--only received\|sent` | restrict to one side of a correspondence |
| `--mode auto\|lexical\|semantic` | select the engines |
| `--prefer recent` | order comparable answers newest first |

`info` reads the package header and requires no key: format, issuer, creation
date, document and passage counts, integrity, signature, language, dates and
calibration.

`--prefer recent` enters as a third ranking fused by rank rather than as a decay
applied to a score. It orders rather than filters: an older document that answers
better is still returned. Where both engines agree on the best passage the date
does not move it; where they disagree, it decides. When the preference cannot be
applied — no meaning index, or no dated passage in the answer — the reply states
so rather than returning silently unchanged.

## Retrieval model

Word matching and dense retrieval answer different questions and are kept side
by side. The lexical index knows that a document contains a term; the dense index
knows that a document means something similar. A query in one language cannot
reach a document in another by word matching, because the words are not shared;
representing meaning is what crosses that boundary.

The two are merged by reciprocal rank rather than by score. A BM25 score is
unbounded and depends on the corpus it was measured in, and a cosine runs from
zero to one and does not; the two cannot be added, and normalising them
introduces a weighting nothing justifies. Rank is what both engines agree on.

Documents are ranked as documents rather than as isolated passages, so a document
that answers in several places is not outranked by one that mentions the terms
once.

## Calibration

A reply states when nothing in the corpus bears on the question, rather than
presenting its nearest passage as an answer. The threshold for that judgement is
a property of the corpus and is measured when the package is built.

An absolute threshold cannot serve: how near a corpus comes to a question it
answers depends on what the corpus contains, so a value set on one collection
marks nothing on another. Each package therefore records its own reach, and the
judgement is made against that.

`pack --focus "<question>"` takes the reach from the questions a package exists
to answer rather than estimating it from the passages. It is the better source
where passages do not resemble questions — a catalogue of abstracts, for example,
where every record shares a rhetorical shape. Repeat the option to give several.
`info` reports which of the two was used.

```
mdcx calibrate corpus.mdcx --key "passphrase" --question "..." --question "..."
```

`calibrate` measures a package that already exists, without its source material:
the measurement needs the vectors, which are in the package, and questions, which
come from outside. The package is rewritten around the changed measurement — same
documents, same passages, same vectors — and the provenance is recorded as
`focus-after` rather than `focus`, since measuring while packing and measuring
afterwards describe the corpus at different moments. A signed package requires its
signing key.

A package built before this measurement existed carries none, and is judged by
the previous thresholds unchanged.

## Vocabulary

```python
archive.vocabulary(connection)           # term frequencies, and the rule
archive.unknown_terms(connection, text)  # terms this corpus has never seen
archive.unfamiliar(connection, text)     # the same, with the share and a flag
```

The index records terms of three characters or more, and one character where the
writing system does not separate words. Absence from the table therefore has two
meanings — the corpus never saw the term, or the index would never have recorded
it — and a caller weighting terms by rarity must distinguish them, since an absent
term takes the maximum weight. `unknown_terms` applies the rule.

Function words are not removed on top of that. Which words carry no information
depends on the question, and `die` in *die casting* or `les` in *Les Misérables*
carry meaning in the language being searched. What mdcx can state is what it never
recorded.

The keys are normalised — folded case, folded accents — so a caller tokenising
with `search.tokenize_text` must normalise before looking a term up.
`unknown_terms` does so.

The word index does not cross languages, while the meaning index does. Asked
across languages, `unknown_terms` returns every term of the text, which describes
the corpus's language rather than its contents. `unfamiliar` reports the share and
a `cross_language` flag; prefer it where the language of the question is not known
to match the corpus.

### Two signals, and their disagreement

`archive.assess(connection, text)` returns both.

A cosine cannot distinguish the senses of a homonym, because a multilingual
embedding places them together: a corpus of algebra can come close to a question
about graph colouring and return passages about the plot of a function. The
literal vocabulary does distinguish them, since the word the question turns on is
absent from it.

`assess` reports both signals and overrules neither. Whether an unfamiliar word
should refuse a query depends on what the query is for, and a word may be
peripheral to it.

## Transcription recovery

A word that optical character recognition transcribed wrongly is a word the index
does not contain, and literal matching cannot return it. The passage remains in
the corpus and the question cannot reach it; the reply is empty, with no error to
indicate why.

Optical recognition confuses characters that are drawn alike. Grouping characters
by shape therefore places a misread word and its original in the same bucket. The
bucket is too coarse to answer with, so it is used as a filter: it proposes
candidates, and edit distance decides among them.

```
mdcx pack --output docs --target corpus.mdcx --key "..." --shapes
mdcx shapes corpus.mdcx --key "..."      # for a package written without it
```

The index is optional because it enlarges the package and serves only corpora
that passed through optical recognition.

Three limits apply. The filter addresses errors of transcription and not of
typing — its advantage comes from the shape of the characters, and so does its
boundary. A substantial share of what it proposes is not a transcription error,
since for Latin script the shape classes are coarse. And it operates only where no
term of the query matched, since one term the corpus does contain is sufficient to
return passages.

Where a passage was located under a different spelling the reply records it, in
`read_as`. Presenting such a passage as a literal match would attribute to the
document a word it does not contain.

The technique is that of character shape codes, described by Spitz at Xerox in the
1990s and applied there over document images to avoid a full recognition pass.
Here it is applied over text already converted, for the errors the conversion left
behind.

## Attachments

A document with `indexed: false` in its front matter is kept in the package —
signed, encrypted, in the same file — and excluded from the index, the vectors and
the passage count. It is intended for material that accompanies a corpus without
being text to search: a certificate, a table of coordinates.

`export` restores it verbatim, and the MCP `document` tool returns it by name,
which is the only way to reach it since it cannot be found by searching.

`pack` reports which document contributed the most passages, and says so when one
contributes a third or more.

## Dates

A package records when each work is from, together with how that was determined.

| Provenance | Meaning |
|---|---|
| `source` | supplied from the publisher or catalogue |
| `sidecar` | supplied by the caller |
| `front-matter` | carried by the document |
| `mtime` | the file's modification time, not the work's |

`mtime` is never used unless `--date-from-mtime` requests it, since it is
available for every file and describes the file rather than the work. Where
nothing reliable is found the date is absent.

The copyright year printed in a document is deliberately not used: a reprint
carries the year of the printing rather than of the edition.

```
mdcx pack --output docs --target corpus.mdcx --key "..." --dates dates.csv
```

Each line of the sidecar is `path,date[,provenance]`. `info` reports how many
documents carry a date and the span they cover, and every passage in a reply
carries `dated` and `dated_from`.

## Correspondence

Where a collection is correspondence, `pack` records which side each document came
from, and `search --only received|sent` restricts the query to one of them. The
classification is taken from the directory structure of the input.

## Incremental packaging

```
mdcx pack --output ./docs --target corpus-2.mdcx --key "..." \
    --multilingual --reuse corpus-1.mdcx
```

A passage whose text has not changed has the same vector. `--reuse` reads the
vectors of an existing package and encodes only what is new. The vectors are read
from a package already encrypted with the same key; no intermediate store is
created, since a vector permits the text it represents to be approximated.

Reuse also carries the calibration forward. A package given its questions with
`--focus` passes them to the package built from it, so a corpus that grows retains
the threshold it was calibrated with. Passing `--focus` again overrides what was
inherited, and the summary records when a calibration was inherited.

Compression and encryption are properties of the whole file, so they cost the same
whether one document was added or the corpus rebuilt. `pack` reports
`seconds_seal` so a caller writing frequently can decide how often to write.

The database is written out as it is built and then compressed, encrypted and
hashed in one pass over it, in blocks. The peak memory of writing a package is
therefore set by the block rather than by the size of the corpus.

## Resident conversion

The conversion engines load once per process. A caller running one process per
document — the usual way to distribute work and isolate failures — pays that load
for every document.

```python
from mdcx.convert import resident

with resident.warm(report=print) as convert:
    for path in queue:
        record = convert(path, output_root)
```

The files written are those `mdcx-convert` writes for that document, chapters
included: a long PDF becomes one Markdown per chapter in a folder of the
document's name, alongside an index Markdown linking them. `split=False` asks
for the document as a single file instead.

`record` is the record `mdcx-convert` writes. Where the document was split it is
the index record, and the chapter records are under `record["chapter_records"]`.
`resident.convert_documents(paths, output_root)` is the same as a generator,
yielding one record per document, and `resident.cost_of_starting()` returns the
seconds spent, so a log can report them. Several such processes each amortise
their own startup.

Handing a whole folder to `mdcx-convert` also amortises the load, at the cost of
the ordering, per-document handling and failure isolation a queue provides.

### Expressions

A lexical index is built out of words, and the rule that decides what a word is
discards the symbols. For prose that is correct. For a corpus interrogated by
statement it removes the content: `pq | b(b+p+q)` and `pq | b(b-p-q)` differ by
one sign and reduce to the same terms, and asked on their own they reduce to no
terms at all and retrieve nothing.

`--expressions` keeps them beside the words, as their own index. A token that
mixes symbols with alphanumerics is an expression; prose yields none. A question
carrying an expression the corpus states is then answered by it, and one
carrying an expression the corpus does not state is answered with nothing rather
than with the passages that merely discuss the subject. Lookup is exact, since
anything looser returns the confusion the index exists to remove.

An expression containing a space is not recovered: deciding where a formula ends
inside a sentence is a different problem, and guessing would fill the index with
fragments of prose.

### Quotations across a cut

A quoted phrase often starts in one passage and ends in the next. `--quotes
document`, the default, keeps the normalised text of every document and looks in
it: there is no limit to how long a quotation may be, and it is a second copy of
the corpus — measured on one package, a third of the file.

`--quotes boundary` indexes the join between consecutive passages instead, in a
table that keeps the index and not the text. A quotation spanning one cut is
found by phrase match; one longer than the join is not, which the whole copy
would still find. The header records which shape a package has.

## MCP server

```
MDCX_FILE=/path/to/corpus.mdcx
MDCX_KEY=passphrase

python -m mdcx.mcp_server
```

The key is supplied through the environment rather than on the command line, where
it would be visible in the process table.

`MDCX_FILE` and `MDCX_KEY` accept several packages, separated by the platform's
path separator or by commas, queried as one corpus. One key serves all of them, or
one key per package in the same order.

Three tools are exposed:

| Tool | Returns |
|---|---|
| `search` | passages answering a question, with provenance |
| `info` | the corpus record without querying it |
| `document` | the full text of one document, by name or portable path |

A `search` reply carries the passages with their source and rank, and reports what
it could not do: `warning` when nothing in the corpus bears on the question,
`read_as` when a passage was located under a different spelling, `prefer_applied`
when a requested preference could not be honoured, and `unknown_terms` naming, per
package, words of the question that package has never seen.

Where several packages are served, `similarity` is computed over all of them
together while `answerable_at` is the lowest of their calibrated reaches, so the
threshold in force comes from the narrowest package rather than from the one a
passage came from. `answerable_at_by_package` reports each.

The server ends its own process when its client disconnects, and releases the
embedding model after `MDCX_IDLE_UNLOAD_MINUTES` of inactivity — 30 by default,
`0` to disable. The model reloads on the next question.

## Sources

mdcx converts, packages and answers. It does not fetch, and depends on no network
of its own. A catalogue is a plugin, declared through the `mdcx.sources` entry
point group and meeting the contract in `mdcx.sources`:

```toml
[project.entry-points."mdcx.sources"]
oapen = "my_package.oapen:Catalogue"
```

Keeping adapters out is deliberate. What appears to be a simple HTTP client is
not: catalogues differ in how a download is reached, how paging is expressed, and
how identifiers are formed, and that knowledge belongs with whoever holds it.
Answering questions over an existing package requires none of this.

What does not depend on any particular catalogue is provided:

```
python -m mdcx.sources --check <name>
```

```python
sources.looks_like(data, "pdf")   # by signature, not by declared type
sources.identify(data)            # what the bytes are instead
sources.patiently(call)           # retries RateLimited, honours Retry-After
sources.conforms(source)          # the conformance check, as a function
```

`Candidate.download` records which server a file would be fetched from, where the
catalogue knows it without a further request, and `Candidate.host` reads it back.
The conformance check reports when one host accounts for more than half of the
candidates.

`tests/test_sources_kit.py` contains a reference source of about twenty lines,
implemented against no network.

## Portable paths

Every document carries a portable path of the form `@/folder/document.md`,
relative to the root of the collection. Absolute paths are not stored, so a
package is readable on a machine whose directory layout differs. `export`
reconstructs the directory structure from them.

## Signing

```
mdcx keygen
mdcx pack --output ./docs --target corpus.mdcx --key "..." \
    --issuer "Organisation" --signing-key <hex private key>
mdcx verify corpus.mdcx --public-key <hex public key>
```

Signing is Ed25519 over the digest of the encrypted body together with the header,
so altering either invalidates the signature. Verification requires only the
public key, not the passphrase.

## Encryption

The package body is encrypted with AES-256-GCM, with the key derived by scrypt
from the passphrase and a per-package salt. The header — format, issuer, counts,
integrity digest, signature, language, dates and calibration — is outside the
encrypted body and readable without the key, so a recipient can decide whether a
package is worth opening.

An empty passphrase is refused at packing time. Supply the passphrase through
`MDCX_KEY`, `--key-file`, or `--key -` to read it from standard input; `--key`
places it in the process table for the duration of the command.

## Package format

A `.mdcx` file is a magic number, a JSON header, and an encrypted body. The body is
a compressed SQLite database holding:

| Table | Contents |
|---|---|
| `document` | one row per document: name, portable path, origin, dates, normalised text |
| `passage` | one row per passage, with its position in its document |
| `passage_fts` | the full-text index, declared over `passage` |
| `passage_vector` | the embedding of each passage, when meaning was indexed |
| `term_shape` | the transcription-recovery index, when built |
| `attachment` | documents kept with the corpus and excluded from it |
| `df` | document frequency per term |
| `meta` | corpus record: language, calibration, model, counts |

Embeddings are stored in half precision and renormalised when read. A package
written by an earlier version omits the tables added since, and every entry point
that reads one checks first, so an older package continues to answer.

## Limitations

- Conversion fidelity is measured against the text a document exposes. A document
  that exposes none cannot be verified, and is marked as such.
- Retrieval across languages requires the `multilingual` extra and its model.
- A package written before 1.27.0 carries the tokenizer it was built with,
  which cut scripts that write their vowels as combining marks at every mark.
  Such a package opens and answers unchanged; rebuilding it indexes those words
  whole.
- The transcription-recovery filter addresses errors of transcription, not of
  typing, and a substantial share of what it proposes is not an error.
- A package is decrypted into memory in full. A corpus larger than available
  memory is held as several packages queried as one, and the server opens each
  only when a query reaches it; `info` reports which are open and what they
  hold. Writing a package does not have this bound: it is built and sealed in
  blocks.
- Calibration thresholds are measured per corpus. A package built before that
  measurement existed is judged by fixed thresholds.

## Tests

```
pytest
```

| File | Covers |
|---|---|
| `test_formats.py` | engine selection and escalation, per format |
| `test_package_identity.py` | the container: schema, encryption, signing, integrity |
| `test_relevance.py` | ranking, and which passages an answer draws on |
| `test_multilingual.py`, `test_encoder.py` | encoding, fusion and cross-language retrieval |
| `test_answer_quality.py` | calibration thresholds and the warning |
| `test_dates.py` | dates, their provenance and the recency preference |
| `test_shapekey.py` | transcription recovery and its boundaries |
| `test_expressions.py` | statements the word rule cannot tell apart |
| `test_quotations.py` | a quotation that crosses the cut between passages |
| `test_sources_kit.py` | the source contract and its conformance check |
| `test_card_sizing.py` | device detection, lane sizing and turns on the card |
| `test_resident_output.py` | what the resident converter writes, chapters included |
| `test_packing_cost.py` | the memory of packing and the cost of reading the original |
| `test_deep_documents.py` | documents deeper than the interpreter's recursion limit |
| `test_closed_package.py` | calibrating and extending an existing package |
| `test_incremental.py` | reuse of vectors between packages |
| `test_server_leaves.py` | server lifetime and idle release |
| `test_describes_itself.py` | that every published field and argument is documented |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issue templates for defects and
measurements are under `.github/ISSUE_TEMPLATE`.

## Security

See [SECURITY.md](SECURITY.md) for supported versions and how to report a
vulnerability.

## Releases

Released to [PyPI](https://pypi.org/project/mdcx/), to the MCP registry as
`io.github.jorgell23-sys/markdown-document-search`, and archived on Zenodo.

[CHANGELOG.md](CHANGELOG.md) records what changed in each release.

## Authorship

Jorge Ellena G. See [NOTICE](NOTICE) for third-party components and their
licences.

## Citation

To cite mdcx in academic work, use [CITATION.cff](CITATION.cff) or the DOI:

```
https://doi.org/10.5281/zenodo.22015991
```

## Licence

Apache License 2.0. See [LICENSE](LICENSE).
