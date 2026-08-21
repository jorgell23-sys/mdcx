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


def _split_setting(value: str) -> list[str]:
    """Split a setting that may name several packages.

    The separator is the one the platform uses for lists of paths, so a Windows
    path keeps its drive letter. A comma is also accepted, since it is what
    people write.
    """
    partes: list[str] = []
    for trozo in value.split(","):
        partes.extend(t for t in trozo.split(os.pathsep) if t.strip())
    return [t.strip() for t in partes if t.strip()]


def _open_packages() -> list[dict]:
    """Open every configured package once and reuse them.

    Decryption and decompression take a fraction of a second and need not be
    repeated per query. Each database is held in memory, so an open package
    leaves no plaintext copy on disk.
    """
    if "packages" in _STATE:
        return _STATE["packages"]

    ajuste = os.environ.get("MDCX_FILE", "").strip()
    claves_ajuste = os.environ.get("MDCX_KEY", "")
    if not ajuste:
        raise RuntimeError("MDCX_FILE is not set: provide the path to the .mdcx package.")
    if not claves_ajuste:
        raise RuntimeError("MDCX_KEY is not set: the package is encrypted.")

    rutas = _split_setting(ajuste)
    claves = _split_setting(claves_ajuste)
    # One key serves every package, which is the ordinary case; several keys are
    # matched to the packages in order.
    if len(claves) == 1:
        claves = claves * len(rutas)
    if len(claves) != len(rutas):
        raise RuntimeError(
            f"MDCX_KEY names {len(claves)} keys for {len(rutas)} packages: "
            "give one key for all of them, or one key per package in the same order.")

    paquetes: list[dict] = []
    for ruta, clave in zip(rutas, claves):
        destino = Path(ruta)
        if not destino.is_file():
            raise RuntimeError(f"Package not found: {destino}")
        connection, header = archive.open_package(destino, clave)
        paquetes.append({"name": destino.name, "path": destino,
                         "connection": connection, "header": header})

    _STATE["packages"] = paquetes
    _STATE["connection"] = paquetes[0]["connection"]
    _STATE["header"] = paquetes[0]["header"]
    return paquetes


def _connection():
    """The first package, for the operations that read a single record."""
    return _open_packages()[0]["connection"]



def search_packages(query: str, limit: int = 5,
                    only: str | None = None) -> list[dict]:
    """Search every configured package and return one ranked list.

    Scores from different packages are computed over different corpus
    statistics -- the frequency of a term depends on the corpus it is measured
    in -- so they cannot be compared to one another. Position within each
    package can, which is what reciprocal rank merges.
    """
    paquetes = _open_packages()
    if len(paquetes) == 1:
        return archive.query(paquetes[0]["connection"], query, limit=limit, only=only)

    from .semantic import fuse

    por_clave: dict = {}
    listas: list[list] = []
    for paquete in paquetes:
        lista = []
        for item in archive.query(paquete["connection"], query, limit=limit,
                                  only=only):
            item = dict(item)
            item["package"] = paquete["name"]
            clave = (paquete["name"], item["document"], item["passage"][:120])
            por_clave[clave] = item
            lista.append(clave)
        listas.append(lista)
    return [por_clave[c] for c in fuse(listas)[:limit]]


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
            "When the package was built with meaning indexed, a query reaches "
            "documents written in other languages as well; use info to see "
            "whether it was. Otherwise matching is by word, and the query "
            "should be written in the language of the documents."
        ),
    )
    async def search(query: str, limit: int = 5,
                     direction: str | None = None) -> dict:
        """Return passages answering the query.

        query: the question, phrased as it would be asked of a person.
        limit: number of passages to return, between 1 and 20.
        direction: "received" or "sent" to restrict the search; omit for all.
        """
        top = max(1, min(int(limit), 20))
        scope = (direction or "").lower().strip() or None
        if scope not in ("received", "sent"):
            scope = None
        results = search_packages(query, top, scope)
        respuesta = {
            "query": query,
            "found": len(results),
            "passages": [
                {
                    "document": item["document"],
                    "direction": item["source"].lower(),
                    "path": item["pseudopath"],
                    "score": item.get("score"),
                    "text": item["passage"],
                    **({"package": item["package"]} if "package" in item else {}),
                }
                for item in results
            ],
        }
        if not results:
            aviso = archive.language_mismatch(connection, query)
            if aviso:
                respuesta["hint"] = aviso
        return respuesta

    def _cross_language() -> str:
        """How this server can answer, in the terms that change the answer."""
        connection = _connection()
        if not archive.has_vectors(connection):
            return "no: this package indexes words only"
        if not archive._semantic_ready(connection):
            return ("not in this installation: the package indexes meaning, but "
                    "the multilingual extra is missing or set to another model")
        return "yes: a query in one language reaches documents in the others"

    def _describe(paquete: dict) -> dict:
        header = paquete["header"]
        return {
            "package": paquete["name"],
            "format": f"{header.get('file_format')} v{header.get('version')}",
            "issuer": header.get("issuer") or "(not declared)",
            "created_utc": header.get("created_utc"),
            "documents": header.get("documents"),
            "passages": header.get("passages"),
            "language": header.get("language") or "(not detected)",
            "integrity": "intact" if header.get("_intact") else "ALTERED",
            "conversion": header.get("conversion", {}),
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
        paquetes = _open_packages()
        if len(paquetes) == 1:
            descripcion = _describe(paquetes[0])
            descripcion.pop("package")
            descripcion["cross_language_search"] = _cross_language()
            return descripcion

        # Several packages are queried as one corpus, so the totals describe the
        # whole of it and the list says where each part comes from.
        partes = [_describe(p) for p in paquetes]
        return {
            "packages": len(partes),
            "documents": sum(p["documents"] or 0 for p in partes),
            "passages": sum(p["passages"] or 0 for p in partes),
            "integrity": ("intact" if all(p["integrity"] == "intact" for p in partes)
                          else "ALTERED"),
            "cross_language_search": _cross_language(),
            "each": partes,
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
        paquetes = _open_packages()
        for paquete in paquetes:
            connection = paquete["connection"]
            row = connection.execute(
                "SELECT d.name, d.pseudopath, d.source, "
                "       group_concat(p.text, char(10) || char(10)) "
                "FROM document d JOIN passage p ON p."
                + archive.document_column(connection) + " = d.id "
                "WHERE d.name = ? OR d.pseudopath = ? "
                "GROUP BY d.id ORDER BY p.position LIMIT 1",
                (name, name)).fetchone()
            if row:
                salida = {"found": True, "document": row[0], "path": row[1],
                          "direction": row[2], "text": row[3] or ""}
                if len(paquetes) > 1:
                    salida["package"] = paquete["name"]
                return salida
        donde = "this corpus" if len(paquetes) == 1 else f"any of the {len(paquetes)} packages"
        return {"found": False,
                "message": f"No document named {name!r} in {donde}."}

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
