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

"""Pseudopaths: identificadores portables, independientes de la ubicacion real de la carpeta.

Un pseudopath es la ruta POSIX relativa a la raiz del arbol convertido, con prefijo `@/`.
Ejemplo:  @/Recibido/01 Contrato/Anexo 1.1 Alcance del trabajo.md

Sirve para que el indice siga siendo valido si la carpeta Output se mueve, se renombra
o se comparte en otro equipo: nunca contiene letras de unidad ni rutas absolutas.
"""
from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

PSEUDO_PREFIX = "@"

# Extensiones que sabemos convertir, agrupadas por familia de motor.
SUPPORTED = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
    ".xlsx": "xlsx",
    ".xlsm": "xlsx",
    ".xls": "xlsx",
    ".pptx": "pptx",
    ".txt": "text",
    ".md": "text",
    ".csv": "text",
    ".html": "html",
    ".htm": "html",
}


def to_pseudopath(rel: Path) -> str:
    """Convierte una ruta relativa en pseudopath portable."""
    return f"{PSEUDO_PREFIX}/" + rel.as_posix()


def from_pseudopath(pseudo: str, root: Path) -> Path:
    """Resuelve un pseudopath contra una raiz concreta (donde sea que este hoy)."""
    if not pseudo.startswith(PSEUDO_PREFIX + "/"):
        raise ValueError(f"pseudopath invalido: {pseudo!r}")
    return root / pseudo[len(PSEUDO_PREFIX) + 1:]


def file_digest(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 del archivo fuente: permite detectar cambios y saltar reconversiones."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _norm(text: str) -> str:
    """Normaliza para comparar nombres: NFC evita que tildes descompuestas rompan matches."""
    return unicodedata.normalize("NFC", text)


@dataclass
class Job:
    """Una unidad de conversion: un archivo fuente y su destino .md espejo.

    Un documento extenso se divide en varios Job, uno por capitulo. Todos comparten el
    mismo `source` y se distinguen por `page_range`; `parent_target` apunta al .md indice
    que los agrupa, de modo que la correspondencia con el original sigue siendo trazable.
    """

    source: Path                 # ruta absoluta al original
    rel_source: Path             # ruta relativa a Input/
    rel_target: Path             # ruta relativa a Output/ (termina en .md)
    kind: str                    # familia de motor: pdf | docx | xlsx | text | ...
    size: int
    digest: str = ""
    meta: dict = field(default_factory=dict)
    page_range: tuple | None = None      # (primera, ultima) 1-based, si es un capitulo
    chapter_title: str = ""              # titulo del capitulo dentro del documento
    chapter_index: int = 0               # orden del capitulo, empezando en 1
    parent_target: Path | None = None    # .md indice del documento completo

    @property
    def is_chapter(self) -> bool:
        return self.page_range is not None

    @property
    def pseudopath(self) -> str:
        return to_pseudopath(self.rel_target)

    @property
    def source_pseudopath(self) -> str:
        return to_pseudopath(self.rel_source)


def plan_jobs(input_root: Path) -> tuple[list[Job], list[Path]]:
    """Recorre Input/ y planifica un .md por archivo, resolviendo colisiones de nombre.

    Dos fuentes distintas en la misma carpeta (informe.pdf e informe.docx) producirian
    el mismo informe.md; en ese caso se desambigua con el sufijo de extension original
    para que la correspondencia siga siendo 1:1 y reversible.
    """
    jobs: list[Job] = []
    skipped: list[Path] = []
    by_target: dict[str, list[Job]] = {}

    for path in sorted(input_root.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        kind = SUPPORTED.get(ext)
        if kind is None:
            skipped.append(path)
            continue

        rel_source = path.relative_to(input_root)
        rel_target = rel_source.with_suffix(".md")
        job = Job(
            source=path,
            rel_source=rel_source,
            rel_target=rel_target,
            kind=kind,
            size=path.stat().st_size,
        )
        by_target.setdefault(_norm(rel_target.as_posix()).lower(), []).append(job)
        jobs.append(job)

    for group in by_target.values():
        if len(group) < 2:
            continue
        for job in group:
            tag = job.source.suffix.lower().lstrip(".")
            stem = job.rel_source.stem
            job.rel_target = job.rel_source.with_name(f"{stem}__{tag}.md")

    return jobs, skipped


# Carriles de ejecucion. Cada documento se asigna al recurso que le conviene, y ambos
# carriles avanzan a la vez: el motor con modelos ocupa la GPU mientras los extractores
# deterministas saturan la CPU, en vez de competir por los mismos nucleos.
LANE_GPU = "gpu"   # requiere modelos de layout/tablas/OCR
LANE_CPU = "cpu"   # extractor determinista, sin modelos


def classify_lane(job: "Job") -> str:
    """Decide el carril de un documento con una inspeccion barata del original.

    No lee el texto completo: solo consulta cuantas paginas tiene y si trae imagenes,
    que es lo unico necesario para saber si el motor pesado aporta algo. Un plano o un
    archivo de texto no ganan nada con el modelo de layout, asi que van directo a CPU.
    """
    if job.kind in ("text", "html"):
        return LANE_CPU
    if job.kind != "pdf":
        return LANE_GPU  # DOCX/XLSX/PPTX: el motor estructurado aporta tablas y titulos

    try:
        from . import pdf as _pdf

        doc = _pdf.leer_documento(job.source)
        try:
            pages = len(doc)
            muestra = min(pages, 3)
            chars = sum(len(_pdf.texto_de_pagina(doc[i]).strip())
                        for i in range(muestra))
            imagenes = sum(_pdf.contar_imagenes(doc[i]) for i in range(muestra))
        finally:
            doc.close()
    except Exception:
        return LANE_GPU

    cpp = chars / max(1, muestra)
    if cpp < 60:
        # Sin capa de texto: solo el OCR puede leerlo, y el OCR necesita el carril pesado.
        return LANE_GPU
    if pages <= 2 and imagenes >= 1:
        return LANE_CPU  # plano o diagrama: medido, el motor pesado empeora el resultado
    return LANE_GPU


def estimated_cost(job: "Job") -> float:
    """Coste relativo de convertir un documento, para ordenar la cola.

    El motor con modelos trabaja pagina por pagina, asi que el numero de paginas domina
    el tiempo. Se usa para despachar primero lo barato: si dos documentos de 300 paginas
    encabezan la cola, ocupan todos los procesos durante media hora y los 50 documentos
    de una pagina que venian detras esperan sin necesidad. Ordenando de menor a mayor,
    el grueso del lote queda listo enseguida y los pesados terminan al final.
    """
    # Un capitulo cuesta lo que su tramo, no lo que el documento del que salio: medirlo
    # por el original haria que los 27 capitulos de un PDF de 300 paginas se ordenaran
    # todos como si fueran de 300, anulando el criterio de despachar primero lo barato.
    if job.page_range is not None:
        return float(job.page_range[1] - job.page_range[0] + 1)

    paginas = 0
    if job.kind == "pdf":
        # contar_paginas devuelve cero si el archivo no se puede leer, y ese cero se
        # resuelve mas abajo con el tamano en disco. No hace falta envolverlo.
        from . import pdf as _pdf

        paginas = _pdf.contar_paginas(job.source)
    if not paginas:
        # Sin recuento de paginas, el tamaño en disco es la mejor aproximacion disponible.
        return job.size / 100_000
    return float(paginas)


def make_chapter_jobs(job: "Job", chapters: list) -> list["Job"]:
    """Expande un documento extenso en un Job por capitulo.

    Los capitulos se depositan en una carpeta con el nombre del documento, y el .md que
    llevaria el documento completo pasa a ser el indice que los enlaza. Asi el espejo de
    la carpeta de entrada sigue teniendo una entrada por documento original.
    """
    carpeta = job.rel_target.with_suffix("")   # .../Documento/  (sin extension)
    salida: list["Job"] = []
    for cap in chapters:
        salida.append(
            Job(
                source=job.source,
                rel_source=job.rel_source,
                rel_target=carpeta / f"{cap.slug()}.md",
                kind=job.kind,
                size=job.size,
                page_range=(cap.first_page, cap.last_page),
                chapter_title=cap.title,
                chapter_index=cap.index,
                parent_target=job.rel_target,
            )
        )
    return salida
