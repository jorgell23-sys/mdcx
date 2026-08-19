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

"""Retira del Markdown el andamiaje que no es contenido, antes de escribirlo.

El motor de conversion emite las tablas con las celdas rellenas de espacios para que las
columnas queden alineadas en el archivo, la fila separadora con decenas de guiones por
columna, la rejilla completa aunque haya columnas enteras sin un solo dato, y marcadores
propios como <!-- image --> donde no pudo transcribir. Nada de eso proviene del documento
original ni lo lee nadie: es la mitad del peso del corpus.

Medido sobre los 187 archivos convertidos, retirarlo ahorra un 46,0 % de los tokens sin
perder una sola palabra ni mover el resultado de ninguna de las 20 consultas del cliente.
Ese fue el nivel 14 del banco de pruebas, el mas agresivo que superaba las dos condiciones
a la vez; los niveles que ademas quitaban numeros de pagina o lineas repetidas ahorraban
mas pero borraban contenido legitimo, y quedaron descartados.

Las mismas funciones las usa tool/comprimir.py, para que lo que se mide en el banco y lo
que escribe el convertidor sean exactamente el mismo codigo.

La compactacion se aplica archivo por archivo y se comprueba en el momento: si el texto
resultante perdiera cualquier palabra respecto del que entro, se descarta y se escribe el
original. Un ahorro nunca justifica perder una palabra, que es la condicion del proyecto.
"""
from __future__ import annotations

import re

from . import verify

# Campos de la cabecera que se conservan. formato_origen: parece prescindible y no lo es:
# es el campo por el que las herramientas del proyecto distinguen la procedencia de cada
# archivo. Retirarlo ahorraba una linea y dejaba ciega a la busqueda, una degradacion que
# no aparece al medir contenido porque el contenido seguia intacto.
CAMPOS_CABECERA = (
    "origen:",
    "pseudopath_origen:",
    "formato_origen:",
    "paginas:",
    "estado_verificacion:",
    "capitulo_titulo:",
)


def compactar_tablas(texto: str) -> str:
    """Quita el relleno de espacios que alinea verticalmente las columnas."""
    salida = []
    for linea in texto.splitlines():
        s = linea.strip()
        if s.startswith("|") and s.endswith("|") and s.count("|") >= 2:
            celdas = [c.strip() for c in s[1:-1].split("|")]
            salida.append("| " + " | ".join(celdas) + " |")
        else:
            salida.append(linea)
    return "\n".join(salida)


def compactar_separadores(texto: str) -> str:
    """Reduce la fila separadora de tabla a los tres guiones por columna que exige Markdown."""
    def _sep(m: re.Match) -> str:
        columnas = m.group(0).strip().strip("|").split("|")
        return "|" + "|".join("---" for _ in columnas) + "|"

    return re.sub(r"(?m)^[ \t]*\|[\s\-:|]+\|[ \t]*$", _sep, texto)


def quitar_marcadores(texto: str) -> str:
    """Elimina los marcadores del propio conversor, del estilo <!-- image -->.

    No provienen del documento: son senales sobre lo que el motor no pudo transcribir. Los
    marcadores que si llevan informacion, como el numero de pagina, no se tocan.
    """
    return re.sub(r"(?m)^[ \t]*<!--\s*(image|imagen)\s*-->[ \t]*\n?", "", texto)


def colapsar_vacias(texto: str) -> str:
    """Deja como maximo una linea en blanco seguida y quita los espacios al final de linea."""
    texto = re.sub(r"[ \t]+$", "", texto, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", texto)


def comprimir_cabecera(texto: str) -> str:
    """Reduce la cabecera de metadatos a los campos que sirven para citar y auditar."""
    if not texto.startswith("---"):
        return texto
    partes = texto.split("---", 2)
    if len(partes) < 3:
        return texto
    cab, cuerpo = partes[1], partes[2]
    lineas = [l for l in cab.splitlines() if l.strip().startswith(CAMPOS_CABECERA)]
    if not lineas:
        return texto
    return "---\n" + "\n".join(lineas) + "\n---" + cuerpo


def quitar_columnas_vacias(texto: str) -> str:
    """Elimina de cada tabla las columnas que no contienen ningun dato.

    Casi la mitad de las celdas del corpus estan vacias: el motor conserva la rejilla
    completa del original aunque una columna entera este en blanco. La decision se toma
    sobre la tabla completa, nunca sobre filas sueltas, y una columna se conserva en cuanto
    alguna fila que no sea la separadora tenga algo escrito.
    """
    lineas = texto.splitlines()
    salida: list[str] = []
    i = 0
    while i < len(lineas):
        if not lineas[i].strip().startswith("|"):
            salida.append(lineas[i])
            i += 1
            continue
        j = i
        while j < len(lineas) and lineas[j].strip().startswith("|"):
            j += 1
        tabla = lineas[i:j]
        filas = [[c.strip() for c in l.strip()[1:-1].split("|")] for l in tabla]
        ancho = max((len(f) for f in filas), default=0)
        es_sep = [bool(re.fullmatch(r"[\s\-:]*", "".join(f))) for f in filas]
        util = [any(len(f) > c and f[c] and not es_sep[k] for k, f in enumerate(filas))
                for c in range(ancho)]
        if not any(util) or all(util):
            salida.extend(tabla)
        else:
            for k, f in enumerate(filas):
                celdas = [f[c] if len(f) > c else "" for c in range(ancho) if util[c]]
                if es_sep[k]:
                    salida.append("|" + "|".join("---" for _ in celdas) + "|")
                else:
                    salida.append("| " + " | ".join(celdas) + " |")
        i = j
    return "\n".join(salida)


# Orden acumulativo del nivel adoptado. Se aplica siempre completo: cada paso supone que
# el anterior ya normalizo las tablas.
PASOS = (
    compactar_tablas,
    compactar_separadores,
    quitar_marcadores,
    colapsar_vacias,
    comprimir_cabecera,
    quitar_columnas_vacias,
)


def compactar(texto: str) -> tuple[str, bool]:
    """Compacta el Markdown y comprueba, en el mismo acto, que no perdio contenido.

    Devuelve el texto y si la compactacion se aplico. La comprobacion es la misma que usa
    la verificacion de fidelidad: se cuentan las palabras del cuerpo antes y despues, como
    multiconjunto, ignorando el andamiaje de Markdown que precisamente se esta retirando.
    Si falta cualquier palabra, se devuelve el texto original sin tocar.
    """
    try:
        comprimido = texto
        for paso in PASOS:
            comprimido = paso(comprimido)
        # Un unico salto final. Sin esto, aplicar la compactacion dos veces daba resultados
        # que diferian en ese caracter, y la operacion dejaba de ser idempotente: relanzar
        # una conversion reescribia archivos que en realidad no habian cambiado.
        comprimido = comprimido.rstrip("\n") + "\n"
    except Exception:  # noqa: BLE001
        # Un texto con una estructura inesperada no debe impedir que el archivo se escriba.
        return texto, False

    antes = verify.tokenize(verify.strip_markdown_noise(_cuerpo(texto)))
    despues = verify.tokenize(verify.strip_markdown_noise(_cuerpo(comprimido)))
    if _falta_algo(antes, despues):
        return texto, False
    return comprimido, True


def _cuerpo(texto: str) -> str:
    """Devuelve el texto sin la cabecera de metadatos.

    La cabecera se compara aparte porque la compactacion la reduce a proposito: incluirla
    en el recuento haria fallar la comprobacion por campos que no son contenido.
    """
    if texto.startswith("---"):
        partes = texto.split("---", 2)
        if len(partes) >= 3:
            return partes[2]
    return texto


def _falta_algo(antes: list[str], despues: list[str]) -> bool:
    """Indica si alguna palabra del texto original no sobrevive, contando repeticiones."""
    from collections import Counter

    faltan = Counter(antes)
    faltan.subtract(Counter(despues))
    return any(n > 0 for n in faltan.values())
