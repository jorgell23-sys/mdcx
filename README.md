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
| `pip install "mdcx[convert]"` | document conversion (Docling, PyTorch) | ~1.4 GB |
| `pip install "mdcx[all]"` | everything, including OCR | ~1.5 GB |

Conversion is what pulls in the heavy dependencies. Someone who receives an
`.mdcx` file and only needs to query it installs neither Docling nor PyTorch.

## Converting a collection

```
pip install "mdcx[convert]"
mdcx-convert --input ./Documents --output ./Documents_md
```

The output mirrors the input directory structure, adds a global index, and
records for each file the coverage achieved against its original.

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

The suite covers hostile inputs: empty and corrupted files, names in other
alphabets, malformed queries including SQL injection attempts, truncated and
tampered packages, and compaction against content loss.

## Language

Retrieval is lexical: a query matches words that appear in the documents. It
therefore works in whatever language a corpus is written in — English, German,
French, Spanish, Portuguese, Italian and any other the tokenizer segments — and
it cannot cross between languages. A Spanish query finds nothing in an English
corpus, however well indexed, because the words are not there.

The package records the predominant language of a corpus when it is built, and
`info` reports it. When a query returns nothing and none of its terms appear in
the index, the result says so and names the corpus language, so an empty answer
can be told apart from material the corpus does not hold.

Earlier versions shipped a glossary of 83 Spanish terms and advertised Spanish
queries against English documents. Those terms all came from piping engineering
and project management; measured against a corpus of mathematics, biology and
history the glossary contributed nothing, and the claim it backed failed on
eleven of twelve queries. Removing it left retrieval unchanged even on the
engineering corpus it was written for: 19 of 20 queries still find the right
document in the top five results.

A glossary remains available for anyone who wants one, as a decision of theirs
rather than an assumption of the package:

```python
from mdcx import search
search.GLOSSARY = search.load_glossary("my-glossary.json")
```

The file maps a term to its equivalents: `{"caneria": ["piping", "pipe"]}`.

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
