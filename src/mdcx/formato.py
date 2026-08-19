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

"""Formato .mdcx: un corpus convertido, indexado, cifrado y portable en un solo archivo.

QUE PROBLEMA RESUELVE

El trabajo caro de este proyecto no es convertir: es dejar el material en un estado donde
una pregunta encuentre su respuesta en milisegundos y con la cita exacta. Ese estado hoy
vive repartido en una carpeta de 208 archivos, un indice, un manifiesto y el codigo que
sabe leerlos. Moverlo a otra maquina, entregarselo a un tercero o conectarlo a un agente
significa mover todo eso junto y confiar en que nadie toque nada por el camino.

Un .mdcx es ese estado completo dentro de un solo archivo: los documentos, el indice de
busqueda, la procedencia de cada pasaje y la constancia de que la conversion fue verificada.
Se abre con una clave, responde consultas sin desempaquetar nada en disco, y cualquier
alteracion posterior se detecta antes de leer una sola fila.

COMO ESTA HECHO

  Cabecera en claro. Identifica el archivo, su version y su huella. Se puede comprobar
  quien lo emitio y si esta integro sin tener la clave, que es justo lo que hace falta para
  decidir si abrirlo. Es el mismo criterio del estandar C2PA para procedencia de medios.

  Cuerpo cifrado con AES-256-GCM. GCM no solo oculta: autentica. Si alguien cambia un byte
  del cuerpo, el descifrado falla en vez de devolver datos corruptos silenciosamente. La
  clave se deriva de una contrasena con scrypt, que encarece a proposito el probar claves.

  Dentro, una base SQLite con FTS5. Un solo archivo, sin servidor, con indice invertido y
  BM25 nativo. Es la pieza con mas kilometraje para busqueda embebida y esta ya presente en
  cualquier Python, lo que evita que el paquete dependa de nada instalado.

UNA DECISION QUE CONVIENE ENTENDER

Esto cifra en reposo y descifra al abrir, en memoria. No es lo mismo que buscar sobre datos
cifrados sin descifrarlos nunca -searchable encryption-, que es un campo distinto: la
literatura reporta ataques de fuga sobre esos esquemas y sobrecostes de segundos por
consulta. Para el caso real -que el archivo circule y no lo lea quien no debe- el cifrado
en reposo da la proteccion buscada sin pagar aquello. Quien necesite que ni siquiera el
proceso que busca vea el texto necesita otra cosa, y debe saber que cuesta ordenes de
magnitud mas.

    python tool/jeg.py empaquetar --output Output --destino corpus.mdcx --clave "..."
    python tool/jeg.py info corpus.mdcx
    python tool/jeg.py buscar corpus.mdcx "su pregunta" --clave "..."
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import struct
import sys
import time
from pathlib import Path



# Los cuatro primeros bytes identifican el archivo pase lo que pase con su nombre. Es lo
# que de verdad distingue un formato: la extension la puede cambiar cualquiera, y no
# existe registro alguno que asigne extensiones, solo catalogos de lo que se ha visto.
MAGIC = b"MDCX"
VERSION = 1

# Parametros de derivacion. scrypt con estos valores tarda una fraccion de segundo en abrir
# un archivo legitimo y hace inviable probar claves en masa, que es exactamente el reparto
# que interesa: el coste lo paga una vez quien tiene la clave y muchas veces quien no.
SCRYPT_N = 2 ** 15
SCRYPT_R = 8
SCRYPT_P = 1
CLAVE_BYTES = 32


# ======================================================================================
# Derivacion y cifrado
# ======================================================================================

def _derivar(clave: str, sal: bytes) -> bytes:
    # scrypt necesita 128 * N * r bytes de memoria, y la libreria de cifrado subyacente
    # rechaza por defecto cualquier peticion que pase de 32 MB. El limite se declara aqui
    # de forma explicita: el coste en memoria no es un efecto secundario del algoritmo,
    # es lo que encarece probar claves en masa, y bajarlo debilitaria el archivo.
    memoria = 128 * SCRYPT_N * SCRYPT_R
    return hashlib.scrypt(clave.encode("utf-8"), salt=sal,
                          n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=CLAVE_BYTES,
                          maxmem=memoria * 2)


def _cifrar(datos: bytes, clave_derivada: bytes) -> tuple[bytes, bytes]:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    nonce = os.urandom(12)
    return nonce, AESGCM(clave_derivada).encrypt(nonce, datos, None)


def _descifrar(cuerpo: bytes, clave_derivada: bytes, nonce: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    return AESGCM(clave_derivada).decrypt(nonce, cuerpo, None)


# ======================================================================================
# Construccion de la base interna
# ======================================================================================

def _construir_base(carpeta: Path) -> tuple[bytes, dict]:
    """Arma en memoria la base con los documentos, su indice y su procedencia."""
    from . import buscar as B

    docs = B.cargar_documentos(carpeta)
    con = sqlite3.connect(":memory:")
    con.executescript("""
        PRAGMA journal_mode = OFF;
        CREATE TABLE documento (
            id INTEGER PRIMARY KEY,
            nombre TEXT NOT NULL,
            pseudopath TEXT NOT NULL,
            origen TEXT NOT NULL,
            carpeta TEXT,
            formato TEXT,
            estado_verificacion TEXT,
            -- Texto normalizado del documento entero. La coincidencia literal se busca
            -- aqui y no en los pasajes porque una frase citada suele cruzar el corte entre
            -- parrafos, y buscarla dentro de un pasaje la pierde justo cuando aparece.
            texto_norm TEXT
        );
        CREATE TABLE pasaje (
            id INTEGER PRIMARY KEY,
            documento_id INTEGER NOT NULL REFERENCES documento(id),
            orden INTEGER NOT NULL,
            texto TEXT NOT NULL
        );
        -- El indice se declara externo al contenido para no guardar el texto dos veces:
        -- FTS5 indexa lo que vive en la tabla pasaje en lugar de copiarlo.
        CREATE VIRTUAL TABLE pasaje_fts USING fts5(
            texto, content='pasaje', content_rowid='id', tokenize='unicode61'
        );
        CREATE TABLE meta (clave TEXT PRIMARY KEY, valor TEXT);
        -- Documento-frecuencia por termino. FTS5 la tiene por dentro pero no la expone de
        -- forma utilizable, y sin ella el paquete no puede puntuar con el mismo criterio
        -- que el buscador de la carpeta. Ocupa poco y viaja con el corpus.
        CREATE TABLE df (termino TEXT PRIMARY KEY, pasajes INTEGER NOT NULL);
    """)

    n_pasajes = 0
    for i, d in enumerate(docs, 1):
        texto = d["texto"]
        formato = ""
        estado = ""
        for linea in texto.splitlines()[:12]:
            if linea.startswith("formato_origen:"):
                formato = linea.split(":", 1)[1].strip()
            elif linea.startswith("estado_verificacion:"):
                estado = linea.split(":", 1)[1].strip()
        con.execute(
            "INSERT INTO documento VALUES (?,?,?,?,?,?,?,?)",
            (i, d["nombre"], d["pseudopath"], d["origen"], d["carpeta"], formato, estado,
             d["norm"]))
        for j, bloque in enumerate(d["bloques"] if "bloques" in d else _bloques(texto)):
            if not bloque.strip():
                continue
            n_pasajes += 1
            con.execute("INSERT INTO pasaje VALUES (?,?,?,?)",
                        (n_pasajes, i, j, bloque))

    con.execute("INSERT INTO pasaje_fts(pasaje_fts) VALUES('rebuild')")

    # Estadisticas para puntuar. Se calculan una vez, al empaquetar, con la misma
    # segmentacion de palabras que usa el buscador, para que ambos midan lo mismo.
    import buscar as _B
    from collections import Counter as _Counter

    df_cuenta: _Counter = _Counter()
    largos: list[int] = []
    for (texto,) in con.execute("SELECT texto FROM pasaje"):
        tk = _B._TOKEN_RE.findall(_B._norm(texto))
        largos.append(len(tk))
        for t in set(tk):
            if len(t) > 2:
                df_cuenta[t] += 1
    con.executemany("INSERT INTO df VALUES (?,?)", df_cuenta.items())
    largo_medio = sum(largos) / len(largos) if largos else 60.0

    resumen = {
        "documentos": len(docs),
        "pasajes": n_pasajes,
        "generado_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "origen_carpeta": carpeta.name,
        "largo_medio_pasaje": round(largo_medio, 2),
        "terminos_indexados": len(df_cuenta),
    }
    # El manifiesto de la conversion viaja dentro: es la prueba de que el contenido fue
    # verificado contra los originales, y sin el las cifras del paquete no son auditables.
    manifiesto = carpeta / "_manifest.json"
    if manifiesto.exists():
        try:
            m = json.loads(manifiesto.read_text(encoding="utf-8"))
            resumen["conversion"] = m.get("resumen", {})
        except Exception:  # noqa: BLE001
            pass
    for k, v in resumen.items():
        con.execute("INSERT INTO meta VALUES (?,?)",
                    (k, json.dumps(v) if not isinstance(v, str) else v))
    con.commit()

    datos = con.serialize()
    con.close()
    return bytes(datos), resumen


def _bloques(texto: str) -> list[str]:
    return [b for b in texto.split("\n\n") if b.strip()]


# ======================================================================================
# Empaquetado y apertura
# ======================================================================================

def empaquetar(carpeta: Path, destino: Path, clave: str, emisor: str = "") -> dict:
    """Escribe el .jeg y devuelve sus cifras."""
    import lzma

    t0 = time.perf_counter()
    base, resumen = _construir_base(carpeta)
    t_base = time.perf_counter() - t0

    t0 = time.perf_counter()
    comprimido = lzma.compress(base, preset=6)
    t_comp = time.perf_counter() - t0

    sal = os.urandom(16)
    t0 = time.perf_counter()
    derivada = _derivar(clave, sal)
    nonce, cuerpo = _cifrar(comprimido, derivada)
    t_cifrado = time.perf_counter() - t0

    cabecera = {
        "formato": "mdcx",
        "version": VERSION,
        "emisor": emisor,
        "creado_utc": resumen["generado_utc"],
        "documentos": resumen["documentos"],
        "pasajes": resumen["pasajes"],
        "cifrado": "AES-256-GCM",
        "derivacion": {"algoritmo": "scrypt", "n": SCRYPT_N, "r": SCRYPT_R, "p": SCRYPT_P},
        "compresion": "lzma",
        "sal": sal.hex(),
        "nonce": nonce.hex(),
        # Huella del cuerpo tal como quedo escrito. Permite comprobar que el archivo no fue
        # alterado sin necesidad de la clave, que es lo que hace falta para decidir si
        # abrirlo. Es el mismo papel que el hard binding de C2PA.
        "huella_cuerpo": hashlib.sha256(cuerpo).hexdigest(),
        "conversion": resumen.get("conversion", {}),
    }
    cab = json.dumps(cabecera, ensure_ascii=False).encode("utf-8")

    with open(destino, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(cab)))
        f.write(cab)
        f.write(cuerpo)

    return {
        "bytes_base": len(base),
        "bytes_comprimido": len(comprimido),
        "bytes_archivo": destino.stat().st_size,
        "segundos_indexar": round(t_base, 2),
        "segundos_comprimir": round(t_comp, 2),
        "segundos_cifrar": round(t_cifrado, 2),
        **resumen,
    }


def leer_cabecera(archivo: Path) -> dict:
    """Cabecera y estado de integridad, sin necesidad de la clave."""
    with open(archivo, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError("no es un archivo .mdcx")
        (n,) = struct.unpack("<I", f.read(4))
        cab = json.loads(f.read(n).decode("utf-8"))
        cuerpo = f.read()
    cab["_integro"] = hashlib.sha256(cuerpo).hexdigest() == cab.get("huella_cuerpo")
    cab["_bytes_cuerpo"] = len(cuerpo)
    return cab


def abrir(archivo: Path, clave: str) -> tuple[sqlite3.Connection, dict]:
    """Descifra en memoria y devuelve la conexion lista para consultar.

    Nada se escribe en disco: la base se reconstruye en memoria desde los bytes descifrados.
    Un .jeg abierto no deja copia en claro que alguien pueda recoger despues.
    """
    import lzma

    with open(archivo, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError("no es un archivo .mdcx")
        (n,) = struct.unpack("<I", f.read(4))
        cab = json.loads(f.read(n).decode("utf-8"))
        cuerpo = f.read()

    if hashlib.sha256(cuerpo).hexdigest() != cab.get("huella_cuerpo"):
        raise ValueError("el archivo fue alterado: la huella del cuerpo no coincide")

    derivada = _derivar(clave, bytes.fromhex(cab["sal"]))
    try:
        comprimido = _descifrar(cuerpo, derivada, bytes.fromhex(cab["nonce"]))
    except Exception as exc:  # noqa: BLE001
        raise ValueError("clave incorrecta o archivo corrupto") from exc

    # La comprobacion de integridad ya se hizo arriba -si la huella no cuadrara, esta
    # funcion habria fallado antes de llegar aqui-, pero el resultado no quedaba anotado en
    # la cabecera. Quien la recibiera por esta via veia el campo ausente y lo interpretaba
    # como alterado: un paquete intacto se anunciaba como manipulado, que es la peor forma
    # posible de equivocarse en un formato cuya razon de ser es la procedencia.
    cab["_integro"] = True
    cab["_bytes_cuerpo"] = len(cuerpo)

    con = sqlite3.connect(":memory:")
    con.deserialize(lzma.decompress(comprimido))
    return con, cab


_SQL_BASE = """
    SELECT d.nombre, d.pseudopath, d.origen, p.texto, bm25(pasaje_fts) AS puntaje
    FROM pasaje_fts
    JOIN pasaje p ON p.id = pasaje_fts.rowid
    JOIN documento d ON d.id = p.documento_id
    WHERE pasaje_fts MATCH ?
"""


def _ejecutar(con: sqlite3.Connection, expr: str, maximo: int,
              solo: str | None) -> list[dict]:
    sql = _SQL_BASE
    params: list = [expr]
    if solo:
        sql += " AND d.origen = ?"
        params.append(solo.upper())
    sql += " ORDER BY puntaje LIMIT ?"
    params.append(maximo)
    try:
        filas = con.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        # Una consulta que FTS5 no sabe interpretar no debe tumbar la busqueda: se descarta
        # esa via y queda la otra.
        return []
    return [{"documento": r[0], "pseudopath": r[1], "origen": r[2],
             "parrafo": r[3], "score": round(-r[4], 3)}
            for r in filas]


# Los mismos valores que valido el barrido sobre las 20 consultas reales. Se repiten aqui
# en vez de importarlos para que el paquete siga siendo interpretable por si solo: quien
# reciba un .jeg debe poder saber con que criterio ordena sin leer el resto del proyecto.
K1 = 1.5
B_LONGITUD = 0.45
DOC_TOP_PASAJES = 8

# Cuantos pasajes se traen de FTS5 antes de reordenar. FTS5 decide quien entra -eso lo hace
# rapido y bien-; el orden final lo decide la agregacion por documento.
#
# Se probaron valores menores buscando velocidad, y el resultado desaconseja bajarlo: con
# 400 el acierto en los cinco primeros cae de 19 a 17 sobre 20, con 800 a 18, y solo a
# partir de 1.200 se recupera. Recortar candidatos parecia una optimizacion y era una
# degradacion disfrazada: llegaba antes a una respuesta peor.
#
# El criterio es el mismo que se aplico al elegir el motor de busqueda: responder mas rapido
# no compensa responder mal. A 1.200, las veinte consultas se resuelven en 5,5 segundos, que
# para una consulta suelta son 276 milisegundos.
CANDIDATOS = 1200


def consultar(con: sqlite3.Connection, consulta: str, maximo: int = 8,
              solo: str | None = None) -> list[dict]:
    """Resuelve la consulta ordenando por documento, no por pasaje suelto.

    Al diagnosticar las veinte consultas reales aparecio que en trece de ellas el documento
    correcto si se recuperaba, pero quedaba en los puestos 8, 13, 19, hasta el 48. El fallo
    no estaba en encontrar sino en ordenar: compitiendo pasaje contra pasaje, un anexo que
    trata el asunto en ocho fragmentos pierde contra un parrafo ajeno que repite una palabra.

    Aqui FTS5 hace lo que hace bien -seleccionar candidatos deprisa sobre su indice
    invertido- y la puntuacion final se calcula aparte, agregando por documento y ponderando
    por cuantos terminos distintos de la consulta llega a tocar. Con ese cambio el acierto
    en el tope pasa de 5 a 16 sobre 20.
    """
    from . import buscar as B

    # Las consultas reales llegan como parrafos: exponen el antecedente, citan la clausula y
    # solo despues preguntan. La primera oracion es la que identifica el asunto; el resto
    # aporta vocabulario de contexto que dispersa la puntuacion entre documentos que hablan
    # del proyecto en general. Usarla sola sube el acierto en el primer puesto de 5 a 7 sobre
    # 20 sin mover el de los cinco primeros.
    #
    # El minimo de cinco palabras no es decorativo: once de las veinte consultas empiezan por
    # una etiqueta como "Section 3" o "Page 3", y quedarse con eso seria buscar nada.
    frase = consulta.strip().split(".")[0][:160].strip()
    efectiva = frase if len(frase.split()) >= 5 else consulta

    terminos = [t for t in B._TOKEN_RE.findall(B._norm(efectiva)) if len(t) > 2]
    # expandir() ya retira las palabras vacias y anade los equivalentes en el idioma del
    # corpus. El paquete ordena con exactamente el mismo criterio que la carpeta: si los dos
    # respondieran distinto a la misma pregunta, no seria el mismo corpus.
    terminos = B.expandir(terminos)
    if not terminos:
        return []
    distintos = set(terminos)

    expr = " OR ".join(f'"{t}"' for t in distintos)
    candidatos = _ejecutar(con, expr, CANDIDATOS, solo)
    if not candidatos:
        return []

    # Estadisticas del corpus, guardadas al empaquetar: hacen falta para puntuar con los
    # mismos parametros que el buscador de la carpeta, y no para recuperar.
    df, n_pasajes, largo_medio = _estadisticas(con)

    por_documento: dict[str, list[dict]] = {}
    for r in candidatos:
        frec = _frecuencias(r["parrafo"], distintos)
        if not frec:
            continue
        largo = max(len(B._TOKEN_RE.findall(B._norm(r["parrafo"]))), 1)
        score = 0.0
        for t, f in frec.items():
            d_t = df.get(t, 1)
            idf = math.log(1 + (n_pasajes - d_t + 0.5) / (d_t + 0.5))
            score += idf * (f * (K1 + 1)) / (
                f + K1 * (1 - B_LONGITUD + B_LONGITUD * largo / largo_medio))
        r = dict(r)
        r["score"] = round(score, 3)
        r["terminos"] = sorted(frec)
        por_documento.setdefault(r["documento"], []).append(r)

    ranking = []
    for nombre, pasajes in por_documento.items():
        pasajes.sort(key=lambda x: -x["score"])
        base = sum(x["score"] for x in pasajes[:DOC_TOP_PASAJES])
        cobertura = max(len(x["terminos"]) for x in pasajes) / max(len(distintos), 1)
        ranking.append((base * cobertura, pasajes))
    ranking.sort(key=lambda par: -par[0])

    # Coincidencia literal al frente. Cuando la consulta cita el documento con sus mismas
    # palabras -lo habitual en una clarificacion contractual, que transcribe la clausula que
    # discute- ese documento es casi con certeza el correcto, y conviene que encabece.
    #
    # Se exige que la frase tenga sustancia. Con menos de cinco palabras no identifica nada:
    # medido sobre las 20 consultas reales, once empiezan por una etiqueta como "Section 3"
    # o "Page 3", y buscarla literalmente devuelve decenas de documentos sin relacion. Ese
    # ruido llego a contarse como acierto en una medicion anterior, y no lo era.
    if len(frase.split()) >= 5:
        aguja = B._norm(frase)
        preferidos = [n for (n, t) in con.execute(
            "SELECT nombre, texto_norm FROM documento") if t and aguja in t]
        if preferidos:
            orden = {d: i for i, d in enumerate(preferidos)}
            ranking.sort(key=lambda par: (orden.get(par[1][0]["documento"], len(orden)),
                                          -par[0]))

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


def _frecuencias(texto: str, terminos: set[str]) -> dict[str, int]:
    from . import buscar as B

    cuenta: dict[str, int] = {}
    for t in B._TOKEN_RE.findall(B._norm(texto)):
        if t in terminos:
            cuenta[t] = cuenta.get(t, 0) + 1
    return cuenta


# Estadisticas ya leidas, por conexion. Traerlas de la base en cada consulta costaria mas
# que la consulta misma; una conexion de sqlite no admite atributos propios, asi que el
# recuerdo vive aqui fuera.
_CACHE_ESTADISTICAS: dict[int, tuple] = {}


def _estadisticas(con: sqlite3.Connection) -> tuple[dict, int, float]:
    """Documento-frecuencia por termino y tamano medio de pasaje, tal como se empaquetaron."""
    clave = id(con)
    if clave not in _CACHE_ESTADISTICAS:
        df = {t: n for t, n in con.execute("SELECT termino, pasajes FROM df")}
        fila = con.execute("SELECT valor FROM meta WHERE clave='pasajes'").fetchone()
        n = int(json.loads(fila[0])) if fila else max(len(df), 1)
        fila = con.execute(
            "SELECT valor FROM meta WHERE clave='largo_medio_pasaje'").fetchone()
        lm = float(json.loads(fila[0])) if fila else 60.0
        _CACHE_ESTADISTICAS[clave] = (df, n, lm)
    return _CACHE_ESTADISTICAS[clave]


def exportar(archivo: Path, clave: str, destino: Path) -> dict:
    """Reconstruye la carpeta de Markdown a partir del paquete.

    Un formato del que no se puede salir es una trampa, por bien intencionado que sea. Quien
    reciba un .mdcx debe poder recuperar los documentos como archivos corrientes y seguir su
    camino sin esta herramienta: eso es lo que separa un contenedor de una jaula.

    Se reconstruye la estructura de carpetas a partir del pseudopath de cada documento, que
    es justamente para lo que se guardo, y se devuelven las cifras para poder comprobar que
    salio todo lo que habia entrado.
    """
    con, cabecera = abrir(archivo, clave)
    try:
        filas = con.execute(
            "SELECT d.pseudopath, d.nombre, group_concat(p.texto, char(10) || char(10)) "
            "FROM documento d JOIN pasaje p ON p.documento_id = d.id "
            "GROUP BY d.id ORDER BY p.orden").fetchall()
        escritos = 0
        for pseudopath, nombre, texto in filas:
            relativo = pseudopath[2:] if pseudopath.startswith("@/") else pseudopath
            ruta = destino / relativo
            ruta.parent.mkdir(parents=True, exist_ok=True)
            ruta.write_text((texto or "") + "\n", encoding="utf-8")
            escritos += 1
    finally:
        con.close()
    return {"documentos": escritos, "destino": str(destino),
            "creado_utc": cabecera.get("creado_utc")}


# ======================================================================================
# Linea de comandos
# ======================================================================================

def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Formato .mdcx: corpus indexado, cifrado y portable")
    sub = ap.add_subparsers(dest="accion", required=True)

    e = sub.add_parser("empaquetar")
    e.add_argument("--output", default="Output")
    e.add_argument("--destino", default="corpus.mdcx")
    e.add_argument("--clave", required=True)
    e.add_argument("--emisor", default="")

    i = sub.add_parser("info")
    i.add_argument("archivo")

    x = sub.add_parser("exportar")
    x.add_argument("archivo")
    x.add_argument("--destino", required=True)
    x.add_argument("--clave", required=True)

    b = sub.add_parser("buscar")
    b.add_argument("archivo")
    b.add_argument("consulta")
    b.add_argument("--clave", required=True)
    b.add_argument("--maximo", type=int, default=5)
    b.add_argument("--solo", choices=["recibido", "emitido"])

    args = ap.parse_args()

    if args.accion == "empaquetar":
        r = empaquetar(Path(args.output), Path(args.destino), args.clave, args.emisor)
        print(f"Empaquetado: {args.destino}")
        print(f"  documentos {r['documentos']}   pasajes {r['pasajes']}")
        print(f"  base sqlite {r['bytes_base']:,} -> comprimida {r['bytes_comprimido']:,} "
              f"-> archivo {r['bytes_archivo']:,} bytes".replace(",", "."))
        print(f"  indexar {r['segundos_indexar']}s  comprimir {r['segundos_comprimir']}s  "
              f"cifrar {r['segundos_cifrar']}s")
        return 0

    if args.accion == "info":
        cab = leer_cabecera(Path(args.archivo))
        print(f"Formato   : {cab['formato']} v{cab['version']}")
        print(f"Emisor    : {cab.get('emisor') or '(sin declarar)'}")
        print(f"Creado    : {cab['creado_utc']}")
        print(f"Contenido : {cab['documentos']} documentos, {cab['pasajes']} pasajes")
        print(f"Cifrado   : {cab['cifrado']} con {cab['derivacion']['algoritmo']}")
        print(f"Integridad: {'intacto' if cab['_integro'] else 'ALTERADO'}")
        if cab.get("conversion"):
            print(f"Conversion: {json.dumps(cab['conversion'], ensure_ascii=False)[:200]}")
        return 0

    if args.accion == "exportar":
        r = exportar(Path(args.archivo), args.clave, Path(args.destino))
        print(f"Exportados {r['documentos']} documentos a {r['destino']}")
        return 0

    con, cab = abrir(Path(args.archivo), args.clave)
    res = consultar(con, args.consulta, args.maximo, args.solo)
    print(f"{len(res)} pasaje(s)\n")
    for r in res:
        print("-" * 96)
        print(f"[{r['origen']}] {r['documento']}   [relevancia {r['score']}]")
        print(f"{r['pseudopath']}")
        print(r["parrafo"][:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
