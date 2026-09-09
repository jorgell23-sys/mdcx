# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each entry states what changed and, where a change rests on a measurement, the
measurement. Entries before 1.10.0 are summarised; the full history is in the
[release notes](https://github.com/jorgell23-sys/mdcx/releases) and in the commit
log.

## [Unreleased]

## [1.25.0] — 2026-09-08

### Fixed

- **Regression in 1.24.1.** Emptying the card's lane handed its processes to
  the processor lane, past the ceiling `_lane_sizes` had set for what the card
  can hold resident. Six or seven processes then loaded models onto a card that
  fits four, filled it, and every turn queued behind what was already there:
  three books took 444.7 s where invoking the tool once per document from
  outside took 65. The freed cores are now left unused while anything reaches
  the card, which is what the card being the bound rather than the processor
  count means. With `--no-docling` nothing is resident and the cores are spent
  in full, as before.
- The run reported `up to N computing at a time` from the size of the GPU lane,
  which with that lane empty is zero -- so it said no process could use the
  card while four of them did. It reports the gate, which is what bounds
  simultaneous use, and names the lane the card is reached from.
- A wait for a turn on the card is reported at 30 s with its cause instead of
  only after the full five minutes. Waiting is not itself the fault; five
  minutes of silence per document is how a run that had been sized wrong
  presented itself as a slow one.
- A PDF whose structure is deeper than the interpreter's recursion limit is
  converted on a second attempt, in a thread with a 64 MiB stack and a limit of
  20,000, and the caller's own limit is restored. Measured by a consumer over
  119,235 mathematics PDFs: two in a hundred thousand were coming out with no
  Markdown at all and `RecursionError` in the record, and converted whole at
  that depth. Where it is still not enough the record says
  `failure: "environment"`, because `RecursionError` alone reads as a broken
  file and the material was being discarded.

### Changed

- `pack()` no longer holds the corpus several times over. The database is
  written out page by page instead of serialised into a second copy, and it is
  then compressed, encrypted and hashed in one pass in blocks of 8 MiB. The
  bytes written are the same ones the previous path produced -- verified byte
  for byte -- so the package format does not change and any reader opens it.
- `mdcx.convert.resident` writes what `mdcx-convert` writes, chapters
  included. It returned well-formed records and wrote a 130-page book as one
  Markdown where the command line writes its chapters and an index; for anyone
  converting in order to retrieve passages the chapter is the unit that gets
  cited, and nothing announced the difference. `split=False` asks for the
  unsplit form deliberately.
- Converting a PDF no longer reads it twice. The plain text of each page was
  extracted once for the reference and again by the native engine, through the
  same function; it is now read once and shared, and the Markdown produced is
  identical. Counting embedded images, which answers only whether a PDF of a
  page or two is a diagram, is no longer done on every page of every document.
  Measured over six books, 929 pages: reading the original fell 32.4% and the
  native engine 16.5%, together 23.5% of converting and verifying.
- The record says what reading the original cost (`seconds_reference`) and what
  comparing against it cost (`seconds_verify`), which are optimised in
  different places and could not be told apart.

## [1.24.1] — 2026-09-08

### Changed

- README rewritten as reference documentation, as described below.

## [1.24.0] — 2026-09-08

### Fixed

- `gpu_available()` no longer reports a card where no device is visible.
  `CUDA_VISIBLE_DEVICES=""` leaves `torch.cuda.is_available()` returning `True`
  with `device_count()` at zero; the device count is now required as well.
- The gate bounding concurrent card use is sized by the card rather than by the
  GPU lane. The hybrid engine reaches the model from either lane, so sizing the
  gate by a lane that had been correctly emptied serialised all card work behind
  one permit.
- A lane with no documents no longer starts worker processes.

### Added

- `mdcx.convert.resident`: convert documents one at a time in one process,
  keeping per-document control while paying the model load once.
- The run reports how many documents used the card against how many were
  expected to, so a wrong lane estimate is stated rather than inferred.
- Documents whose extracted text is dense in private-use-area glyphs — the
  delimiters TeX fonts use — are expected to reach the card.
- `CHANGELOG.md`, in Keep a Changelog format.

### Changed

- README rewritten as reference documentation. Performance figures and
  comparative claims are removed from the prose; section headings are labels
  rather than sentences; the register is declarative throughout. The measurements
  that justify a design decision remain in the code comments and in the tests,
  which is where they are checkable.

## [1.23.0] — 2026-09-07

### Added

- Optional shape index (`pack --shapes`, `mdcx shapes`) recovering terms that
  optical recognition transcribed wrongly. Measured: 76.5 % of misread words
  recovered against 0 % for literal matching, at 49.9 % precision.
- `read_as` in the MCP reply, naming what was read as what.
- `tools/derive_shape_table.py`.

### Changed

- `passage.search_text` is no longer stored after the index is built, with the
  database vacuumed so the pages are actually returned.

## [1.22.0] — 2026-08-31

### Fixed

- The MCP server ends its own process when its client goes, rather than
  returning and waiting on non-daemon threads. Ten orphaned servers were
  measured alive at once holding 7.25 GB.

### Added

- The encoder is released after `MDCX_IDLE_UNLOAD_MINUTES` of silence.
- `Candidate.download` and `Candidate.host`, so a caller can tell which server a
  file would come from before fetching it.

## [1.21.1] — 2026-08-30

### Fixed

- `python -m mdcx.sources` reported a correct plugin as broken. Running a module
  with `-m` defines its classes a second time, so `isinstance` compared two
  distinct `Candidate` classes.
- `--help` printed nothing and ran the checker instead.

## [1.21.0] — 2026-08-30

### Added

- `looks_like`, `identify`, `patiently`, `conforms` and
  `python -m mdcx.sources --check`: what a source plugin should not have to
  write again. No adapter and no network.

## [1.20.0] — 2026-08-30

### Fixed

- Cross-language detection no longer rests on a threshold over the share of
  unknown terms — no cut separates the two reasons vocabulary goes missing.
- The English function-word list omitted `a`, `do` and `you`, so an English
  corpus never matched its own queries.

## [1.19.0] — 2026-08-29

### Fixed

- Query vectors are unit length; the excess above a cosine of 1 came from the
  query side, not the stored side.
- `unknown_terms` is reported per package rather than intersected across them.
- `rapidocr-onnxruntime` removed from the extras: it pulled the CPU
  `onnxruntime` over a prepared `onnxruntime-gpu` on every upgrade.

### Added

- `mdcx[all-gpu]` and `mdcx[ocr-gpu]`, without the CPU runtime pinned.
- `answerable_at_by_package` in the MCP reply.

## [1.18.0] — 2026-08-29

### Added

- `assess()`, returning the semantic and lexical signals together without
  overruling either.
- `unfamiliar()`, distinguishing an unfamiliar question from one asked in
  another language.

### Fixed

- Stored vectors are renormalised on read; half precision costs the
  normalisation.

## [1.17.0] — 2026-08-29

### Added

- `mdcx calibrate`: measure a closed package against questions without its
  source material.
- `vocabulary()` and `unknown_terms()`, carrying the rule that produced `df`.
- `indexed: false` keeps a document in the package and out of the index.

## [1.16.0] — 2026-08-29

### Added

- `closeness()` and `answers()`, per package.
- `--fast`: preset 3 against 6 — six times the speed for 38 % more bytes.

### Fixed

- `--reuse` carries the calibration of the package it reuses.

## [1.15.1] — 2026-08-29

### Fixed

- `--prefer` states when it could not be applied instead of applying nothing
  quietly.

## [1.15.0] — 2026-08-29

### Added

- `dated` and `dated_from` per document, and `--prefer recent`.

## [1.14.0] — 2026-08-28

### Added

- Document sampling, packing a single file, and the `mdcx.sources` plugin
  contract.

## [1.13.0] — 2026-08-28

### Added

- `pack --focus`: the threshold for "nothing here is about this" taken from the
  questions a package exists to answer.

## [1.12.0] — 2026-08-28

### Fixed

- Card permits are counted per document rather than per block.

## [1.11.0] — 2026-08-28

### Added

- `--timeout` and `--recycle-after`, so one document the engine cannot finish
  does not hold a worker for the run.

## [1.10.0] — 2026-08-27

### Added

- `answerable_at`: each package measures how near it comes to what it can
  answer, and the warning is judged against that rather than a constant.

## [1.0.0] — 2026-08

First public release: conversion with measured fidelity, the encrypted `.mdcx`
container, lexical and dense retrieval, and the MCP server.

[Unreleased]: https://github.com/jorgell23-sys/mdcx/compare/v1.25.0...HEAD
[1.25.0]: https://github.com/jorgell23-sys/mdcx/releases/tag/v1.25.0
[1.24.1]: https://github.com/jorgell23-sys/mdcx/releases/tag/v1.24.1
[1.24.0]: https://github.com/jorgell23-sys/mdcx/releases/tag/v1.24.0
[1.23.0]: https://github.com/jorgell23-sys/mdcx/releases/tag/v1.23.0
[1.22.0]: https://github.com/jorgell23-sys/mdcx/releases/tag/v1.22.0
[1.21.1]: https://github.com/jorgell23-sys/mdcx/releases/tag/v1.21.1
[1.21.0]: https://github.com/jorgell23-sys/mdcx/releases/tag/v1.21.0
[1.20.0]: https://github.com/jorgell23-sys/mdcx/releases/tag/v1.20.0
[1.19.0]: https://github.com/jorgell23-sys/mdcx/releases/tag/v1.19.0
[1.18.0]: https://github.com/jorgell23-sys/mdcx/releases/tag/v1.18.0
[1.17.0]: https://github.com/jorgell23-sys/mdcx/releases/tag/v1.17.0
[1.16.0]: https://github.com/jorgell23-sys/mdcx/releases/tag/v1.16.0
[1.15.1]: https://github.com/jorgell23-sys/mdcx/releases/tag/v1.15.1
[1.15.0]: https://github.com/jorgell23-sys/mdcx/releases/tag/v1.15.0
[1.14.0]: https://github.com/jorgell23-sys/mdcx/releases/tag/v1.14.0
[1.13.0]: https://github.com/jorgell23-sys/mdcx/releases/tag/v1.13.0
[1.12.0]: https://github.com/jorgell23-sys/mdcx/releases/tag/v1.12.0
[1.11.0]: https://github.com/jorgell23-sys/mdcx/releases/tag/v1.11.0
[1.10.0]: https://github.com/jorgell23-sys/mdcx/releases/tag/v1.10.0
