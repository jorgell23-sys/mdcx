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

"""Division de documentos extensos en capitulos.

Un PDF de 300 paginas plantea dos problemas distintos:

  - De proceso: ocupa un proceso durante media hora mientras el resto de la cola espera.
    Partido en capitulos, los tramos se reparten entre todos los procesos disponibles.
  - De uso: un unico .md de 300 paginas no se puede leer ni navegar. Partido, cada
    capitulo es un documento con entidad propia y el conjunto conserva un indice.

La division sigue el indice embebido del PDF cuando existe, porque son los cortes que
el autor definio. Solo cuando el documento no trae indice se recurre a tramos de tamaño
fijo, que es un corte arbitrario y se rotula como tal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# A partir de este numero de paginas conviene dividir. Por debajo, el documento se
# convierte de una pieza: partirlo agregaria archivos sin mejorar tiempo ni lectura.
SPLIT_THRESHOLD_PAGES = 60

# Tamaño objetivo de cada capitulo. Un tramo mas grande vuelve a bloquear un proceso;
# uno mas chico multiplica archivos y repite la carga de modelos sin ganancia.
TARGET_CHAPTER_PAGES = 30

# Tramo fijo cuando el documento no trae indice embebido.
FALLBACK_BLOCK_PAGES = 25


@dataclass
class Chapter:
    """Un tramo de paginas de un documento, con el titulo que le corresponde."""

    index: int          # orden dentro del documento, empezando en 1
    title: str          # titulo del indice, o rotulo de paginas si no habia indice
    first_page: int     # 1-based, inclusivo
    last_page: int      # 1-based, inclusivo
    from_toc: bool      # True si el corte proviene del indice del documento

    @property
    def pages(self) -> int:
        return self.last_page - self.first_page + 1

    def slug(self) -> str:
        """Nombre de archivo legible y estable para el capitulo."""
        limpio = re.sub(r"[\\/:*?\"<>|\r\n\t]+", " ", self.title).strip()
        limpio = re.sub(r"\s+", " ", limpio)
        if len(limpio) > 70:
            limpio = limpio[:70].rstrip()
        if not limpio:
            limpio = f"Paginas {self.first_page}-{self.last_page}"
        return f"{self.index:02d} - {limpio}"


def _toc_chapters(toc: list, page_count: int) -> list[Chapter]:
    """Construye capitulos a partir del indice embebido.

    Se empieza por el nivel superior. Si ese nivel produce tramos demasiado largos (un
    documento con solo dos o tres secciones enormes), se baja un nivel para obtener
    cortes mas utiles, siempre que el nivel siguiente exista.
    """
    if not toc:
        return []

    niveles = sorted({lv for lv, _t, _p in toc})
    elegido: list[tuple[str, int]] = []

    for nivel in niveles:
        entradas = [(titulo, pagina) for lv, titulo, pagina in toc
                    if lv == nivel and 1 <= pagina <= page_count]
        if not entradas:
            continue
        # Descartar entradas fuera de orden: un indice mal generado produciria tramos
        # negativos y capitulos vacios.
        entradas = sorted(entradas, key=lambda e: e[1])
        promedio = page_count / len(entradas)
        elegido = entradas
        if promedio <= TARGET_CHAPTER_PAGES * 1.6:
            break  # este nivel ya da tramos de tamaño razonable

    if not elegido:
        return []

    capitulos: list[Chapter] = []
    # Si el primer corte no esta en la pagina 1, el material previo (portada, indice,
    # resumen) forma un capitulo inicial para que no quede fuera del resultado.
    if elegido[0][1] > 1:
        capitulos.append(Chapter(1, "Preliminares", 1, elegido[0][1] - 1, True))

    for i, (titulo, pagina) in enumerate(elegido):
        fin = elegido[i + 1][1] - 1 if i + 1 < len(elegido) else page_count
        if fin < pagina:
            continue  # dos entradas en la misma pagina: la siguiente absorbe el tramo
        capitulos.append(Chapter(len(capitulos) + 1, titulo.strip(), pagina, fin, True))

    return [c for c in capitulos if c.pages > 0]


def _block_chapters(page_count: int, block: int = FALLBACK_BLOCK_PAGES) -> list[Chapter]:
    """Tramos de tamaño fijo, para documentos sin indice embebido."""
    capitulos: list[Chapter] = []
    inicio = 1
    while inicio <= page_count:
        fin = min(inicio + block - 1, page_count)
        capitulos.append(
            Chapter(len(capitulos) + 1, f"Paginas {inicio} a {fin}", inicio, fin, False)
        )
        inicio = fin + 1
    return capitulos


def plan_chapters(pdf_path: Path, threshold: int = SPLIT_THRESHOLD_PAGES) -> list[Chapter]:
    """Devuelve los capitulos de un PDF, o lista vacia si no conviene dividirlo."""
    from . import pdf as _pdf

    page_count = _pdf.contar_paginas(pdf_path)
    if page_count < threshold:
        return []
    # El indice llega ya normalizado como (nivel, titulo, pagina 1-based): el modulo de
    # PDF descarta por su cuenta las entradas cuyo destino el archivo no resuelve.
    toc = _pdf.indice(pdf_path)

    capitulos = _toc_chapters(toc, page_count)

    # Un indice con muy pocos cortes no resuelve el problema: seguiria habiendo tramos
    # de cien paginas ocupando un proceso. En ese caso se subdivide cada tramo largo.
    if capitulos:
        refinados: list[Chapter] = []
        for cap in capitulos:
            if cap.pages > TARGET_CHAPTER_PAGES * 2:
                inicio = cap.first_page
                parte = 1
                while inicio <= cap.last_page:
                    fin = min(inicio + TARGET_CHAPTER_PAGES - 1, cap.last_page)
                    titulo = cap.title if parte == 1 else f"{cap.title} (cont. {parte})"
                    refinados.append(
                        Chapter(len(refinados) + 1, titulo, inicio, fin, cap.from_toc)
                    )
                    inicio = fin + 1
                    parte += 1
            else:
                refinados.append(
                    Chapter(len(refinados) + 1, cap.title, cap.first_page, cap.last_page,
                            cap.from_toc)
                )
        return refinados

    return _block_chapters(page_count)


def extract_range(source: Path, first_page: int, last_page: int, destino: Path) -> Path:
    """Escribe un PDF nuevo con el rango de paginas indicado (1-based, inclusivo)."""
    from . import pdf as _pdf

    return _pdf.recortar(source, first_page, last_page, destino)
