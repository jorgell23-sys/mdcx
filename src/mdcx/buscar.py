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

"""Busqueda con cita exacta sobre la carpeta de documentos convertidos.

Responde a la pregunta practica que motiva la conversion: "esto que afirma el cliente,
¿donde esta escrito y con que palabras?". Devuelve el pasaje literal, el documento que lo
contiene, si ese documento es de los recibidos o de los emitidos, y su pseudopath.

    python tool/buscar.py "vendor information" --contexto 2
    python tool/buscar.py "main supports" --solo emitido
    python tool/buscar.py --frases frases.txt --json salida.json

La busqueda es literal y sin modelos: lo que se cita es exactamente lo que dice el
documento, sin intermediacion de nada que pueda reformular el texto.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path



# Palabras: secuencias alfanumericas, incluidas las acentuadas. Es la misma segmentacion que
# usa la verificacion de fidelidad, para que buscar y medir hablen del mismo token.
_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+", re.UNICODE)


def _norm(texto: str) -> str:
    """Normaliza para comparar: sin acentos, sin mayusculas, espacios colapsados.

    Los documentos vienen de PDF, donde el mismo texto aparece con comillas tipograficas,
    guiones largos o espacios dobles segun como se haya maquetado. Comparar en crudo
    haria fallar coincidencias que para un lector son identicas.
    """
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.replace("’", "'").replace("‘", "'")
    t = t.replace("“", '"').replace("”", '"')
    t = t.replace("–", "-").replace("—", "-").replace("‑", "-")
    return re.sub(r"\s+", " ", t)


def cargar_documentos(output_root: Path) -> list[dict]:
    """Lee los Markdown de la carpeta de salida con su procedencia."""
    docs = []
    for p in sorted(output_root.rglob("*.md")):
        rel = p.relative_to(output_root)
        partes = rel.parts
        if partes[0].startswith("_") or rel.name.startswith("00_"):
            continue
        # La primera carpeta indica la procedencia del documento en el intercambio.
        raiz = partes[0].lower()
        if "emitido" in raiz:
            origen = "EMITIDO"
        elif "recibido" in raiz:
            origen = "RECIBIDO"
        else:
            origen = "OTRO"
        try:
            texto = p.read_text(encoding="utf-8")
        except Exception:
            continue
        cuerpo = texto.split("---", 2)[-1] if texto.startswith("---") else texto
        docs.append({
            "path": p,
            "rel": rel.as_posix(),
            "pseudopath": "@/" + rel.as_posix(),
            "origen": origen,
            "nombre": rel.name[:-3],
            "carpeta": rel.parent.as_posix(),
            "texto": cuerpo,
            "norm": _norm(cuerpo),
        })
    return docs


def _parrafos(texto: str) -> list[str]:
    return [b.strip() for b in re.split(r"\n\s*\n", texto) if b.strip()]


def _recortar_tabla(bloque: str, aguja: str) -> str:
    """Si el bloque es una tabla, deja la cabecera y solo las filas que citan el termino.

    Una tabla de un documento tecnico puede ocupar cientos de filas. Devolverla entera como
    "cita" obliga a buscar a mano dentro de la cita, que es justo el trabajo que se quiere
    evitar; devolver la fila sola pierde el significado de las columnas.
    """
    lineas = bloque.splitlines()
    if sum(1 for l in lineas if l.strip().startswith("|")) < 3:
        return bloque
    cabecera = [l for l in lineas[:2]]
    filas = [l for l in lineas[2:] if aguja in _norm(l)]
    if not filas:
        return bloque
    return "\n".join(cabecera + filas)


# Parametros clasicos de BM25. k1 gradua cuanto suma repetir un termino dentro del mismo
# pasaje; b, cuanto penaliza que el pasaje sea largo.
BM25_K1 = 1.5
# La penalizacion por longitud baja de 0,75 a 0,45. El valor clasico esta pensado para
# colecciones de documentos cortos y homogeneos; aqui los documentos que responden son
# justamente los extensos -un plan de ejecucion, un anexo de criterios- y 0,75 los hundia.
# Medido sobre las 20 consultas reales, este cambio junto con la agregacion por documento
# sube el acierto en el tope de 6 a 17 sobre 20.
BM25_B = 0.45

# Cuantos pasajes de un mismo documento suman a su puntuacion. Sin tope, un documento largo
# gana por acumular menciones de paso; con un tope demasiado bajo, se pierde la senal de que
# el documento trata el tema en varios sitios. Ocho fue el valor que eligio la validacion.
DOC_TOP_PASAJES = 8

# Palabras minimas para que una frase merezca buscarse literalmente. Por debajo de esto no
# identifica un documento: lo encuentra en cualquier parte.
MIN_PALABRAS_LITERAL = 5


_CACHE_BM25: dict = {}


def _indice_bm25(docs: list[dict], solo: str | None) -> dict:
    """Prepara, una sola vez por corpus, las magnitudes que necesita la puntuacion.

    El indice se guarda por identidad de la lista de documentos y procedencia: recorrer los
    187 archivos y contar sus terminos en cada consulta multiplica el costo sin cambiar el
    resultado, porque el corpus no varia entre una consulta y la siguiente.
    """
    clave = (id(docs), solo or "todo")
    if clave in _CACHE_BM25:
        return _CACHE_BM25[clave]
    pasajes = []
    for d in docs:
        if solo and d["origen"] != solo.upper():
            continue
        bloques = d.setdefault("bloques", _parrafos(d["texto"]))
        normas = d.setdefault("bloques_norm", [_norm(b) for b in bloques])
        for i, bn in enumerate(normas):
            tk = _TOKEN_RE.findall(bn)
            pasajes.append({"doc": d, "i": i, "frec": Counter(tk), "largo": len(tk)})
    df = Counter()
    for p in pasajes:
        for t in p["frec"]:
            df[t] += 1
    largo_medio = (sum(p["largo"] for p in pasajes) / len(pasajes)) if pasajes else 1.0
    idx = {"pasajes": pasajes, "df": df, "n": len(pasajes), "largo_medio": largo_medio}
    _CACHE_BM25[clave] = idx
    return idx


# --------------------------------------------------------------------------------------
# Puente de vocabulario entre el idioma de la pregunta y el del corpus.
# --------------------------------------------------------------------------------------
# El corpus de este proyecto esta redactado casi por completo en ingles tecnico, mientras
# que las preguntas llegan en espanol. Medido con una consulta real -donde se indica el
# diametro minimo a modelar en 3D de canerias- el buscador no encontraba nada, no porque el
# dato faltara, sino porque ninguna de las palabras de la pregunta aparece en el documento
# que lo contiene: alli dice piping, minimum y modeled. Traducida a los terminos del corpus,
# la misma consulta devuelve el pasaje exacto.
#
# El puente es una tabla del vocabulario del dominio, escrita a mano y verificada contra el
# corpus. No traduce la consulta ni la reescribe: agrega los equivalentes como terminos
# adicionales, de modo que la pregunta original sigue valiendo y solo se amplia el alcance.
# Un termino que no este en la tabla se busca tal como se escribio.
GLOSARIO = {
    "caneria": ["piping", "pipe"],
    "canerias": ["piping", "pipe"],
    "tuberia": ["piping", "pipe"],
    "tuberias": ["piping", "pipe"],
    "diametro": ["diameter", "bore", "nps", "size"],
    "diametros": ["diameter", "bore", "nps", "size"],
    "minimo": ["minimum", "smallest", "above"],
    "minima": ["minimum", "smallest", "above"],
    "maximo": ["maximum", "largest"],
    "maxima": ["maximum", "largest"],
    "modelar": ["modeled", "modelled", "model", "modeling"],
    "modelado": ["modeled", "modelled", "model", "modeling"],
    "modelo": ["model"],
    "plano": ["drawing"],
    "planos": ["drawings"],
    "entregable": ["deliverable"],
    "entregables": ["deliverables"],
    "alcance": ["scope"],
    "plazo": ["schedule", "duration"],
    "plazos": ["schedule", "milestones"],
    "hito": ["milestone"],
    "hitos": ["milestones"],
    "ingenieria": ["engineering"],
    "detalle": ["detail", "detailed"],
    "basica": ["basic"],
    "acero": ["steel"],
    "estructural": ["structural"],
    "estructuras": ["structures", "structural"],
    "civil": ["civil"],
    "electrico": ["electrical"],
    "electrica": ["electrical"],
    "instrumentacion": ["instrumentation"],
    "proceso": ["process"],
    "procesos": ["process"],
    "equipo": ["equipment"],
    "equipos": ["equipment"],
    "soporte": ["support"],
    "soportes": ["supports"],
    "valvula": ["valve"],
    "valvulas": ["valves"],
    "revision": ["review", "revision"],
    "revisiones": ["reviews"],
    "responsabilidad": ["responsibility", "responsible"],
    "contratista": ["contractor"],
    "proveedor": ["supplier", "vendor"],
    "proveedores": ["suppliers", "vendors"],
    "cliente": ["client", "owner"],
    "requisito": ["requirement"],
    "requisitos": ["requirements"],
    "norma": ["standard", "code"],
    "normas": ["standards", "codes"],
    "criterio": ["criteria", "criterion"],
    "criterios": ["criteria"],
    "interfaz": ["interface"],
    "interfaces": ["interfaces"],
    "integracion": ["integration"],
    "pulgada": ["inch", "nps"],
    "pulgadas": ["inches", "nps"],
    "presion": ["pressure"],
    "temperatura": ["temperature"],
    "capacidad": ["capacity"],
    "costo": ["cost"],
    "costos": ["costs"],
    "precio": ["price"],
    "moneda": ["currency"],
    "pago": ["payment"],
    "pagos": ["payments"],
    "garantia": ["warranty", "guarantee"],
    "seguridad": ["safety", "security"],
    "calidad": ["quality"],
    "riesgo": ["risk"],
    "riesgos": ["risks"],
    "reunion": ["meeting"],
    "reuniones": ["meetings"],
    "informe": ["report"],
    "informes": ["reports"],
    "documento": ["document"],
    "documentos": ["documents"],
    "donde": [],
    "indica": [],
    "cual": [],
    "como": [],
    "para": [],
}

# Palabras que arman una pregunta pero no dicen de que trata. Aparecen en practicamente
# todos los documentos, de modo que no distinguen ninguno, y sin embargo suman puntuacion y
# arrastran hacia arriba textos que solo comparten gramatica con la consulta.
#
# Retirarlas fue el cambio de mayor efecto medido: sobre las 20 consultas reales, el acierto
# en los cinco primeros pasa de 17 a 19 sobre 20, y el documento correcto entra en los diez
# primeros en las 20. El puesto medio del acierto baja de 4,5 a 3,0.
#
# La lista se quedo deliberadamente en lo gramatical. Se probo una version mas larga, que
# incluia el vocabulario de tramite del expediente -mentions, please, confirm, section,
# document, clarification-, y empeoraba el resultado a 15 sobre 20: esas palabras si
# discriminan en un corpus contractual, porque no todos los documentos son una clausula o
# una aclaracion. Quitar de mas cuesta mas que quitar de menos.
PALABRAS_VACIAS = {
    "que", "cual", "cuales", "como", "donde", "cuando", "quien", "porque", "cuanto",
    "what", "which", "how", "where", "when", "who", "why",
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "has",
    "have", "will", "shall", "can", "could", "would", "should", "not", "but", "any",
}


def expandir(terminos: list[str]) -> list[str]:
    """Agrega a la consulta los equivalentes del corpus, conservando los originales.

    Antes de expandir se retiran las palabras vacias: las que arman la pregunta sin decir de
    que trata. Verlas desaparecer de la consulta puede inquietar, pero es lo que hace que el
    resto pese. Los terminos vacios declarados en el propio glosario cumplen la misma
    funcion para casos sueltos.
    """
    salida: list[str] = []
    for t in terminos:
        if t in PALABRAS_VACIAS:
            continue
        equivalentes = GLOSARIO.get(t)
        if equivalentes is None:
            salida.append(t)
        elif equivalentes:
            salida.append(t)
            salida.extend(equivalentes)
        # Si la entrada existe y esta vacia, el termino se descarta a proposito.
    # Se eliminan repeticiones conservando el orden, que es el de la pregunta.
    vistos = set()
    return [t for t in salida if not (t in vistos or vistos.add(t))]


def buscar_terminos(docs: list[dict], consulta: str, contexto: int = 1,
                    solo: str | None = None, maximo: int = 12,
                    minimo_terminos: int = 2) -> list[dict]:
    """Busca por terminos sueltos y ordena los pasajes por relevancia BM25.

    La busqueda literal exige que la frase aparezca tal cual y trata todas las palabras por
    igual, de modo que una consulta redactada con otras palabras no encuentra nada y un
    parrafo larguisimo que menciona el termino de paso compite con el que lo desarrolla.
    BM25 pondera cada termino por lo raro que sea en el corpus y normaliza por la longitud
    del pasaje: gana el fragmento breve que concentra los terminos poco frecuentes.
    """
    consulta_tk = [t for t in _TOKEN_RE.findall(_norm(consulta)) if len(t) > 2]
    consulta_tk = expandir(consulta_tk)
    if not consulta_tk:
        return []
    idx = _indice_bm25(docs, solo)
    n, df, lm = idx["n"], idx["df"], idx["largo_medio"]
    distintos = len(set(consulta_tk))

    # Primera vuelta: puntuar cada pasaje por si mismo.
    por_documento: dict[str, list[dict]] = {}
    for p in idx["pasajes"]:
        presentes = [t for t in consulta_tk if p["frec"].get(t)]
        if len(presentes) < min(minimo_terminos, len(consulta_tk)):
            continue
        score = 0.0
        for t in presentes:
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            f = p["frec"][t]
            score += idf * (f * (BM25_K1 + 1)) / (
                f + BM25_K1 * (1 - BM25_B + BM25_B * p["largo"] / lm))
        d, i = p["doc"], p["i"]
        ini = max(0, i - contexto)
        por_documento.setdefault(d["nombre"], []).append({
            "documento": d["nombre"],
            "origen": d["origen"],
            "pseudopath": d["pseudopath"],
            "carpeta": d["carpeta"],
            "score": round(score, 3),
            "terminos": presentes,
            "parrafo": _recortar_tabla(d["bloques"][i], presentes[0] if presentes else ""),
            "pasaje": "\n\n".join(d["bloques"][ini: i + contexto + 1]),
        })

    # Segunda vuelta: decidir el orden por documento, no por pasaje suelto.
    #
    # Competir pasaje contra pasaje respondia mal a la pregunta que de verdad se hace. Quien
    # consulta quiere saber que documento responde; un anexo que trata el asunto en ocho
    # fragmentos distintos perdia contra un parrafo de otro documento que menciona el
    # termino una vez y lo repite. Al diagnosticar las veinte consultas reales, en trece el
    # documento correcto si aparecia, pero en los puestos 8, 13, 19, hasta el 48.
    #
    # El puntaje de un documento suma sus mejores pasajes y se pondera por cuantos terminos
    # distintos de la consulta llega a tocar: responder de verdad implica tocar casi todos,
    # mientras que puntuar alto repitiendo una sola palabra frecuente no es responder.
    ranking = []
    for nombre, pasajes in por_documento.items():
        pasajes.sort(key=lambda r: -r["score"])
        base = sum(r["score"] for r in pasajes[:DOC_TOP_PASAJES])
        cobertura = max(len(r["terminos"]) for r in pasajes) / max(distintos, 1)
        ranking.append((base * cobertura, pasajes))
    ranking.sort(key=lambda par: -par[0])

    # Se devuelven pasajes, como siempre, pero recorriendo los documentos por orden de
    # relevancia y tomando primero el mejor pasaje de cada uno: asi el tope no se agota con
    # ocho fragmentos del mismo documento.
    salida: list[dict] = []
    for vuelta in range(DOC_TOP_PASAJES):
        for puntaje, pasajes in ranking:
            if vuelta < len(pasajes):
                r = dict(pasajes[vuelta])
                r["score_documento"] = round(puntaje, 3)
                salida.append(r)
                if len(salida) >= maximo:
                    return salida
    return salida[:maximo]


def buscar_hibrido(docs: list[dict], consulta: str, contexto: int = 1,
                   solo: str | None = None, maximo: int = 12) -> list[dict]:
    """Ordena por relevancia y solo antepone lo literal cuando la frase identifica algo.

    Este motor antes empezaba siempre por la coincidencia literal, porque con la puntuacion
    antigua rescataba consultas que BM25 no resolvia. Con el ranking por documento esa
    ayuda se volvio del reves: medido sobre las 20 consultas reales, anteponer lo literal
    sin condiciones baja el acierto en los cinco primeros de 19 a 13 sobre 20.

    La razon estaba escondida en los datos. Once de las veinte consultas empiezan citando
    un apartado -"Section 3", "Page 3"-, asi que la frase literal que se buscaba era esa
    etiqueta, que aparece en decenas de documentos sin relacion. Aquellos aciertos no eran
    recuperacion sino coincidencia, y al mejorar el orden pasaron de inflar la cifra a
    desplazar a los documentos que si responden.

    De ahi la condicion: la frase se antepone solo si tiene sustancia suficiente para
    identificar un documento. Cuando la tiene -una clausula transcrita, una frase citada-
    sigue siendo la senal mas fiable que existe, y encabeza.
    """
    frase = consulta.strip().split(".")[0][:160].strip()
    ordenados = buscar_terminos(docs, consulta, contexto, solo, maximo)

    if len(frase.split()) < MIN_PALABRAS_LITERAL:
        return ordenados

    literales = buscar(docs, frase, contexto, solo, maximo)
    if not literales:
        return ordenados

    vistos = {(r["documento"], r["parrafo"][:80]) for r in literales}
    for r in ordenados:
        clave = (r["documento"], r["parrafo"][:80])
        if clave in vistos:
            continue
        vistos.add(clave)
        literales.append(r)
        if len(literales) >= maximo:
            break
    return literales[:maximo]


def buscar(docs: list[dict], frase: str, contexto: int = 1,
           solo: str | None = None, maximo: int = 12) -> list[dict]:
    """Busca una frase y devuelve los pasajes que la contienen, con su procedencia.

    Se compara parrafo por parrafo, cada uno normalizado por separado. La alternativa
    -normalizar el documento entero y despues reubicar la posicion en el original- es
    fragil: la normalizacion cambia la longitud del texto y el calculo proporcional
    termina senalando un parrafo distinto del que contiene la frase.
    """
    aguja = _norm(frase)
    if not aguja:
        return []
    resultados = []
    for d in docs:
        if solo and d["origen"] != solo.upper():
            continue
        bloques = d.setdefault("bloques", _parrafos(d["texto"]))
        normas = d.setdefault("bloques_norm", [_norm(b) for b in bloques])
        for i, bn in enumerate(normas):
            if aguja not in bn:
                continue
            ini = max(0, i - contexto)
            resultados.append({
                "documento": d["nombre"],
                "origen": d["origen"],
                "pseudopath": d["pseudopath"],
                "carpeta": d["carpeta"],
                "parrafo": _recortar_tabla(bloques[i], aguja),
                "pasaje": "\n\n".join(bloques[ini: i + contexto + 1]),
            })
            if len(resultados) >= maximo:
                return resultados
    return resultados


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


    ap = argparse.ArgumentParser(description="Busqueda con cita exacta sobre los .md convertidos.")
    ap.add_argument("frase", nargs="?", help="texto a buscar")
    ap.add_argument("--output", default="Output", help="carpeta con los .md")
    ap.add_argument("--contexto", type=int, default=1, help="parrafos vecinos a incluir")
    ap.add_argument("--solo", choices=["emitido", "recibido"], help="restringir la procedencia")
    ap.add_argument("--maximo", type=int, default=12, help="tope de pasajes")
    ap.add_argument("--frases", help="archivo con una frase por linea")
    ap.add_argument("--json", help="volcar el resultado a un archivo JSON")
    ap.add_argument("--literal", action="store_true",
                    help="exigir solo la frase exacta, sin respaldo por relevancia")
    ap.add_argument("--bm25", action="store_true",
                    help="usar solo la puntuacion por relevancia, sin la coincidencia literal")
    args = ap.parse_args()

    docs = cargar_documentos(Path(args.output))
    if not docs:
        print("No se encontraron documentos convertidos.", file=sys.stderr)
        return 2

    frases = []
    if args.frases:
        frases = [l.strip() for l in Path(args.frases).read_text(encoding="utf-8").splitlines() if l.strip()]
    elif args.frase:
        frases = [args.frase]
    else:
        print("Indique una frase o un archivo con --frases.", file=sys.stderr)
        return 2

    salida = {}
    for f in frases:
        if args.literal:
            res = buscar(docs, f, args.contexto, args.solo, args.maximo)
        elif args.bm25:
            res = buscar_terminos(docs, f, args.contexto, args.solo, args.maximo)
        else:
            res = buscar_hibrido(docs, f, args.contexto, args.solo, args.maximo)
        salida[f] = res
        print("=" * 100)
        print(f"BUSQUEDA: {f}")
        print(f"  {len(res)} pasaje(s)")
        for r in res:
            print("-" * 100)
            marca = f" [score {r['score']}]" if "score" in r else ""
            print(f"  [{r['origen']}] {r['documento']}{marca}")
            print(f"  {r['pseudopath']}")
            print(f"  {r['parrafo'][:1800]}")
        print()

    if args.json:
        Path(args.json).write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Detalle en {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
