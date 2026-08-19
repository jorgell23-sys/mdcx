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

"""Gate de no-perdida: mide cuanto del texto original sobrevivio en el Markdown.

Metodo: se tokeniza el texto de referencia (leido con una libreria distinta a la del
conversor) y el Markdown resultante, y se comparan como multiconjuntos. Comparar
multiconjuntos y no conjuntos importa: si un codigo aparece 12 veces en el original y
solo 3 en el Markdown, eso es perdida real y un conjunto no lo detectaria.

Se reportan dos coberturas:
  - `coverage`: sobre todos los tokens.
  - `numeric_coverage`: solo sobre tokens que contienen digitos (cantidades, tags de
    equipo, codigos de documento, revisiones). En documentos de ingenieria un numero
    perdido cuesta mucho mas que una palabra de relleno, asi que se vigila aparte.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

# Umbrales de aceptacion. Un documento por debajo de FAIL no se considera convertido.
COVERAGE_OK = 0.995
COVERAGE_WARN = 0.970

_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+", re.UNICODE)

# Ruido que introduce el propio formato Markdown y que no existe en el original: la fila
# separadora de una tabla, las marcas de bloque de codigo y los comentarios que agregan los
# conversores (por ejemplo `<!-- image -->`).
#
# Dos precauciones, ambas aprendidas de perdidas reales medidas sobre el corpus:
#
# 1. Los extremos usan [ \t]* y no \s*. Como \s incluye el salto de linea, en modo
#    multilinea el patron se extendia mas alla de su propia linea y arrastraba texto de
#    las vecinas.
# 2. El delimitador de bloque solo se descarta cuando la linea no contiene nada mas. El
#    conversor emite bloques de codigo enteros en una sola linea, del estilo
#    "``` Format: PROY-{Area}-MOD-{NNNN} Examples: ... ```", y descartar esa linea
#    completa se llevaba por delante la convencion de nomenclatura del proyecto: 250
#    palabras que luego figuraban como perdidas sin que ningun anexo pudiera reponerlas.
_MD_NOISE = re.compile(
    r"(?m)^[ \t]*(?:"
    r"\|[-: |]+\|"              # fila separadora de una tabla: |---|---|
    r"|`{3,}[A-Za-z0-9_+.-]*"   # linea que es SOLO el delimitador de bloque, con o sin lenguaje
    r"|<!--[^\n]*?-->"          # marcador del conversor, por ejemplo <!-- image -->
    r")[ \t]*$"
)


def tokenize(text: str) -> Counter:
    text = unicodedata.normalize("NFC", text)
    return Counter(m.group(0).lower() for m in _TOKEN_RE.finditer(text))


def strip_markdown_noise(md: str) -> str:
    """Quita del Markdown lo que agrega el propio formato y no proviene del original."""
    return _MD_NOISE.sub(" ", md)


# Nombre anterior, conservado por compatibilidad con codigo que ya lo usaba.
_strip_markdown_noise = strip_markdown_noise


def compare(reference: str, markdown: str, sample: int = 25) -> dict:
    """Compara referencia contra Markdown y devuelve el veredicto de fidelidad."""
    ref = tokenize(reference)
    got = tokenize(strip_markdown_noise(markdown))

    total = sum(ref.values())
    if total == 0:
        # Sin texto de referencia (PDF puramente rasterizado o archivo vacio):
        # la cobertura no es medible, se decide por presencia de contenido.
        produced = sum(got.values())
        return {
            "measurable": False,
            "status": "no-reference",
            "coverage": None,
            "numeric_coverage": None,
            "ref_tokens": 0,
            "md_tokens": produced,
            "missing_sample": [],
            "note": "el original no expone texto: fidelidad no verificable por tokens",
        }

    matched = sum(min(count, got[token]) for token, count in ref.items())
    coverage = matched / total

    num_ref = {t: c for t, c in ref.items() if any(ch.isdigit() for ch in t)}
    num_total = sum(num_ref.values())
    num_matched = sum(min(c, got[t]) for t, c in num_ref.items())
    numeric_coverage = (num_matched / num_total) if num_total else None

    deficits = [
        (count - got[token], token)
        for token, count in ref.items()
        if got[token] < count
    ]
    deficits.sort(reverse=True)

    if coverage >= COVERAGE_OK:
        status = "ok"
    elif coverage >= COVERAGE_WARN:
        status = "warn"
    else:
        status = "fail"

    # Perder numeros es mas grave: degrada el veredicto aunque el total luzca bien.
    if numeric_coverage is not None and numeric_coverage < COVERAGE_WARN and status == "ok":
        status = "warn"

    return {
        "measurable": True,
        "status": status,
        "coverage": round(coverage, 5),
        "numeric_coverage": round(numeric_coverage, 5) if numeric_coverage is not None else None,
        "ref_tokens": total,
        "md_tokens": sum(got.values()),
        "missing_tokens": sum(d for d, _ in deficits),
        "missing_sample": [t for _, t in deficits[:sample]],
    }
