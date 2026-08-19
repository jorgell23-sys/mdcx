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

"""Acceso a archivos PDF, apoyado en pypdfium2.

POR QUE ESTE MODULO EXISTE

El proyecto leia PDF con PyMuPDF, repartido por cuatro archivos distintos. PyMuPDF es una
biblioteca excelente y su licencia es AGPL-3.0 o comercial: quien distribuye software que la
usa queda obligado a publicar todo su codigo bajo AGPL, y esa obligacion alcanza tambien a
ofrecerlo como servicio en red. Es una decision legitima de sus autores, pero decide por
nosotros que licencia puede tener esta herramienta.

pypdfium2 hace lo mismo que aqui se necesitaba, esta bajo BSD-3-Clause y Apache-2.0, y ya
venia instalado como dependencia de Docling. Cambiar no anadio nada: quito una biblioteca.

Concentrar el acceso en un solo modulo tiene un segundo motivo, mas duradero que el
licenciamiento: la proxima vez que convenga cambiar de motor de PDF, el cambio ocurre aqui
y no en cuatro sitios que hay que encontrar primero.

UNA DIFERENCIA QUE IMPORTA

pypdfium2 usa el sistema de coordenadas del propio formato PDF, con el origen abajo a la
izquierda; PyMuPDF lo daba invertido, con el origen arriba. Reconstruir el orden de lectura
exige tenerlo presente: aqui se ordena por el borde superior descendente y despues por el
izquierdo, que es como se lee una pagina.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Bloque:
    """Un fragmento de texto con su posicion en la pagina."""
    texto: str
    izquierda: float
    arriba: float


def _abrir(ruta: Path):
    import pypdfium2 as pdfium

    return pdfium.PdfDocument(str(ruta))


def contar_paginas(ruta: Path) -> int:
    """Numero de paginas, o cero si el archivo no se puede leer."""
    try:
        doc = _abrir(ruta)
    except Exception:  # noqa: BLE001
        return 0
    try:
        return len(doc)
    finally:
        doc.close()


def texto_de_pagina(pagina) -> str:
    tp = pagina.get_textpage()
    try:
        return tp.get_text_range() or ""
    finally:
        tp.close()


def bloques_de_pagina(pagina) -> list[Bloque]:
    """Fragmentos de texto de una pagina, en orden de lectura.

    pypdfium2 entrega rectangulos de texto sueltos, no parrafos. Se recogen tal cual y se
    ordenan de arriba abajo y de izquierda a derecha, que es lo que hace falta para que un
    plano o un documento a dos columnas se lea en el orden correcto y no salteado.
    """
    tp = pagina.get_textpage()
    bloques: list[Bloque] = []
    try:
        for i in range(tp.count_rects()):
            izq, abajo, der, arriba = tp.get_rect(i)
            texto = (tp.get_text_bounded(izq, abajo, der, arriba) or "").strip()
            if texto:
                bloques.append(Bloque(texto, round(izq, 1), round(arriba, 1)))
    finally:
        tp.close()
    # Mayor coordenada superior primero: en PDF, mas arriba es un valor mas alto.
    bloques.sort(key=lambda b: (-b.arriba, b.izquierda))
    return bloques


def contar_imagenes(pagina) -> int:
    import pypdfium2 as pdfium

    try:
        return sum(1 for o in pagina.get_objects()
                   if o.type == pdfium.raw.FPDF_PAGEOBJ_IMAGE)
    except Exception:  # noqa: BLE001
        return 0


def leer_documento(ruta: Path):
    """Abre el PDF para recorrerlo. El llamador cierra."""
    return _abrir(ruta)


def indice(ruta: Path) -> list[tuple[int, str, int]]:
    """Indice incorporado del PDF como (nivel, titulo, pagina 1-based).

    Se devuelven solo las entradas que apuntan a una pagina concreta: un indice puede
    contener destinos que el archivo no resuelve, y una entrada sin pagina no sirve para
    partir el documento.
    """
    try:
        doc = _abrir(ruta)
    except Exception:  # noqa: BLE001
        return []
    salida: list[tuple[int, str, int]] = []
    try:
        for marca in doc.get_toc():
            try:
                destino = marca.get_dest()
                if destino is None:
                    continue
                pagina = destino.get_index()
                if pagina is None:
                    continue
                titulo = (marca.get_title() or "").strip()
                if titulo:
                    salida.append((marca.level + 1, titulo, pagina + 1))
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        return []
    finally:
        doc.close()
    return salida


def recortar(origen: Path, primera: int, ultima: int, destino: Path) -> Path:
    """Escribe un PDF nuevo con el rango de paginas indicado, 1-based e inclusivo."""
    import pypdfium2 as pdfium

    src = _abrir(origen)
    try:
        out = pdfium.PdfDocument.new()
        out.import_pages(src, list(range(primera - 1, ultima)))
        destino.parent.mkdir(parents=True, exist_ok=True)
        out.save(str(destino))
        out.close()
    finally:
        src.close()
    return destino


# Tolerancia vertical para considerar que dos fragmentos estan en la misma linea. Se expresa
# en puntos tipograficos, la unidad del PDF: dos puntos es menos que el alto de cualquier
# letra, asi que agrupa lo que esta a la misma altura sin unir renglones distintos.
TOLERANCIA_LINEA = 2.0

# Separacion vertical a partir de la cual se considera que empieza otro parrafo. Por debajo
# de esto las lineas son consecutivas del mismo bloque de texto.
SALTO_PARRAFO = 14.0


def parrafos_de_pagina(pagina) -> list[str]:
    """Texto de la pagina agrupado en lineas y parrafos, en orden de lectura.

    pypdfium2 entrega rectangulos de texto sueltos -en una pagina corriente hay decenas-, no
    parrafos. Volcarlos uno por linea produciria un Markdown troceado en fragmentos de dos
    palabras, ilegible y peor para buscar, porque cada fragmento competiria por separado.

    Se reconstruye en dos pasos: primero se juntan los fragmentos que estan a la misma altura
    -son la misma linea partida en trozos-, y despues las lineas seguidas se unen en un
    parrafo hasta que aparece un salto vertical grande, que es donde el documento cambia de
    bloque.
    """
    bloques = bloques_de_pagina(pagina)
    if not bloques:
        return []

    lineas: list[tuple[float, str]] = []
    actual: list[Bloque] = []
    for b in bloques:
        if actual and abs(actual[-1].arriba - b.arriba) <= TOLERANCIA_LINEA:
            actual.append(b)
            continue
        if actual:
            lineas.append((actual[0].arriba,
                           " ".join(x.texto for x in sorted(actual, key=lambda y: y.izquierda))))
        actual = [b]
    if actual:
        lineas.append((actual[0].arriba,
                       " ".join(x.texto for x in sorted(actual, key=lambda y: y.izquierda))))

    parrafos: list[str] = []
    acumulado: list[str] = []
    anterior: float | None = None
    for arriba, texto in lineas:
        if anterior is not None and (anterior - arriba) > SALTO_PARRAFO:
            if acumulado:
                parrafos.append(" ".join(acumulado))
            acumulado = []
        acumulado.append(texto)
        anterior = arriba
    if acumulado:
        parrafos.append(" ".join(acumulado))
    return parrafos


def parrafos_rapidos(pagina) -> list[str]:
    """Parrafos de la pagina a partir del texto plano, sin consultar rectangulos.

    parrafos_de_pagina pide el texto de cada rectangulo por separado, lo que reconstruye
    con fidelidad documentos a dos columnas pero se vuelve carisimo cuando la pagina tiene
    cientos de fragmentos: en un documento de 307 paginas con tablas densas costaba 315
    segundos, casi un segundo por pagina, frente a 3,4 segundos que tarda leer el texto
    plano completo.

    Aqui se pide el texto de la pagina de una vez, en el orden que declara el propio PDF, y
    se separa en parrafos por los saltos de linea. Se pierde la capacidad de reordenar
    columnas, y a cambio se gana un orden de magnitud. Es el compromiso correcto para el
    motor nativo, que actua como red de seguridad: la fidelidad del contenido se mide contra
    ese mismo texto plano, y cuando el layout importa de verdad quien convierte es Docling.
    """
    texto = texto_de_pagina(pagina)
    if not texto.strip():
        return []
    parrafos: list[str] = []
    actual: list[str] = []
    for linea in texto.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        l = linea.strip()
        if l:
            actual.append(l)
        elif actual:
            parrafos.append(" ".join(actual))
            actual = []
    if actual:
        parrafos.append(" ".join(actual))
    return parrafos
