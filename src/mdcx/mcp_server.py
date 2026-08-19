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

"""Model Context Protocol server exposing an ``.mdcx`` corpus to agents.

An agent answering questions about a document collection can either receive the
documents in its context window, which is expensive and bounded, or query a
component that already knows where each item is. This server provides the second
option: it receives a question, searches locally, and returns only the passages
that answer it, each with its provenance.

Configuration is supplied through environment variables so that the key is not
passed on the command line:

    MDCX_FILE=/path/to/corpus.mdcx
    MDCX_KEY=package-key

    python -m mdcx.mcp_server
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from . import archive

_STATE: dict = {}


def _connection():
    """Open the package once and reuse it.

    Decryption and decompression take a fraction of a second and need not be
    repeated per query. The database is held in memory, so an open package
    leaves no plaintext copy on disk.
    """
    if "connection" in _STATE:
        return _STATE["connection"]

    path = os.environ.get("MDCX_FILE", "").strip()
    key = os.environ.get("MDCX_KEY", "")
    if not path:
        raise RuntimeError("MDCX_FILE is not set: provide the path to the .mdcx package.")
    target = Path(path)
    if not target.is_file():
        raise RuntimeError(f"Package not found: {target}")
    if not key:
        raise RuntimeError("MDCX_KEY is not set: the package is encrypted.")

    connection, header = archive.open_package(target, key)
    _STATE["connection"] = connection
    _STATE["header"] = header
    return connection


def create_server():
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(
        name="mdcx",
        title="Queryable document corpus",
        instructions=(
            "Queries a converted and verified document collection. Use search to "
            "locate the passages that answer a question: it returns the verbatim "
            "text together with the source of each passage, so the source can be "
            "cited rather than recalled. Use info to learn what the corpus "
            "contains before querying it."
        ),
    )

    @server.tool(
        name="search",
        title="Search the corpus",
        description=(
            "Returns the passages that answer a query, each with its source "
            "document, portable path and whether it was sent or received. "
            "Queries may be written in Spanish even when the documents tore in "
            "English."
        ),
    )
    async def search(query: str, limit: int = 5,
                     direction: str | None = None) -> dict:
        """Return passages answering the query.

        query: the question, phrased as it would be asked of a person.
        limit: number of passages to return, between 1 and 20.
        direction: "received" or "sent" to restrict the search; omit for all.
        """
        connection = _connection()
        top = max(1, min(int(limit), 20))
        scope = (direction or "").lower().strip() or None
        if scope not in ("received", "sent"):
            scope = None
        results = archive.query(connection, query, limit=top, only=scope)
        return {
            "query": query,
            "found": len(results),
            "passages": [
                {
                    "document": item["document"],
                    "direction": item["source"].lower(),
                    "path": item["pseudopath"],
                    "score": item.get("score"),
                    "text": item["passage"],
                }
                for item in results
            ],
        }

    @server.tool(
        name="info",
        title="Corpus information",
        description=(
            "Describes the package: number of documents, creation date, issuer "
            "and the fidelity with which it was converted from the originals."
        ),
    )
    async def info() -> dict:
        """Return the corpus record without querying it."""
        _connection()
        header = _STATE.get("header", {})
        return {
            "format": f"{header.get('file_format')} v{header.get('version')}",
            "issuer": header.get("issuer") or "(not declared)",
            "created_utc": header.get("created_utc"),
            "documents": header.get("documents"),
            "passages": header.get("passages"),
            "integrity": "intact" if header.get("_intact") else "ALTERED",
            "conversion": header.get("conversion", {}),
        }

    @server.tool(
        name="document",
        title="Read a full document",
        description=(
            "Returns the complete text of one document, identified by name or by "
            "portable path. Use it only when passages are insufficient: a full "
            "document may span tens of thousands of tokens."
        ),
    )
    async def document(name: str) -> dict:
        """Return the full text of one document in the corpus."""
        connection = _connection()
        row = connection.execute(
            "SELECT d.name, d.pseudopath, d.source, "
            "       group_concat(p.text, char(10) || char(10)) "
            "FROM document d JOIN passage p ON p." + archive.document_column(connection) + " = d.id "
            "WHERE d.name = ? OR d.pseudopath = ? "
            "GROUP BY d.id ORDER BY p.position LIMIT 1",
            (name, name)).fetchone()
        if not row:
            return {"found": False,
                    "message": f"No document named {name!r} in this corpus."}
        return {"found": True, "document": row[0], "path": row[1],
                "direction": row[2], "text": row[3] or ""}

    return server


def main() -> int:
    try:
        _connection()
    except Exception as exc:  # noqa: BLE001
        print(f"Cannot open corpus: {exc}", file=sys.stderr)
        return 2

    try:
        server = create_server()
    except Exception as exc:  # noqa: BLE001
        print(f"Cannot start server: {exc}", file=sys.stderr)
        return 2

    header = _STATE.get("header", {})
    print(f"mdcx: {header.get('documents')} documents ready.", file=sys.stderr)
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
