# Releasing mdcx

A release goes to four places. They are not interchangeable, and a release that
reaches only some of them is a release that reports its version differently
depending on where it is asked.

| Channel | What it carries | How it gets there |
|---|---|---|
| PyPI | the wheel and the source distribution — what `pip install` fetches | `twine upload` |
| MCP Registry | metadata only, so MCP clients can discover the server | `mcp-publisher publish` |
| GitHub Releases | the tag, the notes, and the artefacts | `gh release create` |
| Zenodo | the archived copy and the DOI | automatically, from the GitHub release |

Zenodo is the only one that happens on its own, and only because the repository
is connected to it. The concept DOI `10.5281/zenodo.22015991` always resolves to
the newest version, so it is the one to cite.

## The version lives in three files

A release begins by moving all of them together. They disagree silently, and
each is read by a different audience:

    pyproject.toml    version = "1.7.0"          what pip reports
    server.json       "version" twice            the registry entry and its package reference
    CITATION.cff      version, date-released     what a citation quotes

`src/mdcx/__init__.py` deliberately does **not** carry a version. It reads the
installed metadata instead, because a number written in two places is a number
that will disagree with itself — this one had already drifted three releases
behind what the package actually was.

## Checklist

1. **Bump** the three files above. `date-released` in `CITATION.cff` is the day
   of the release, not the day the work started.
2. **Test.** `python -m pytest tests/ -q`, with the extras installed, so nothing
   skips silently. Continuous integration runs the same suite on Python 3.11 and
   3.13 across Linux and Windows, and has caught things one machine cannot.
3. **Build.** `python -m build` then `python -m twine check dist/*`. The check
   is what says the README will render on PyPI rather than appear as raw text.
4. **Upload.** `python -m twine upload dist/*`, with `__token__` as the username
   and the API token as the password.
5. **Verify against what was published**, not against the working copy: install
   the version from PyPI into a clean interpreter and run the commands. A local
   tree passing proves the tree, not the artefact.
6. **Register.** `mcp-publisher login github` then `mcp-publisher publish`.
7. **Release on GitHub.** `gh release create` with the tag and the notes. Zenodo
   picks it up from there.

The index can lag. A version that uploads successfully and does not appear in
the simple index for a few minutes is usually CDN caching rather than a rejected
upload — that has happened here and resolved itself.

## A published version cannot be replaced

It can be deleted from PyPI, but the number can never be reused. If 1.7.0 has a
defect, the fix is 1.7.1. This is why a rehearsal on TestPyPI is worth the extra
minutes for anything that changes packaging rather than code:

    python -m twine upload --repository testpypi dist/*

TestPyPI is a separate service with its own account and its own token.

## Ownership of the registry name

The registry verifies ownership by looking for a marker in the published package
description. The README carries

    <!-- mcp-name: io.github.jorgell23-sys/markdown-document-search -->

at the top, and `server.json` declares the same name. Both must match, and the
marker must already be present **in the published version** — adding it
afterwards would need a further release.

The `io.github.jorgell23-sys/*` namespace is granted by authenticating with that
GitHub account, so there is no separate ownership claim to file.

The registry is in preview; breaking changes or data resets may still occur.

## First-time setup

Only needed once, and only these two steps need a person.

**PyPI account and token.** Register at https://pypi.org/account/register/;
two-factor authentication is mandatory. Create an API token at
https://pypi.org/manage/account/token/. Scope it to the `mdcx` project — an
account-wide token is only necessary before the project exists. The token is
shown once and begins with `pypi-`.

**GitHub authentication** for `mcp-publisher login github` and for `gh`, which
use a device flow: the command prints a code, and a person enters it in the
browser. The code expires; if it does, ask for another rather than reusing it.

Tokens belong outside the repository. Check `.gitignore` before putting a
credential anywhere near the working tree, and never in a tracked file — a file
listing ignore rules does not ignore itself.
