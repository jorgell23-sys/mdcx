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

"""Servidor MCP: expone un corpus .mdcx como herramienta para agentes.

QUE HACE

Un agente que necesita responder sobre un expediente documental tiene dos caminos. Puede
recibir los documentos enteros en su contexto, que es caro y ademas topa con el limite de la
ventana; o puede preguntar a algo que ya sepa donde esta cada cosa. Este servidor es la
segunda opcion: recibe la pregunta, busca en la maquina y devuelve solo los pasajes que la
responden, cada uno con su procedencia.

La diferencia, medida sobre un expediente real de 99 documentos: responder leyendo los
originales cuesta 2.265.488 tokens de modelo; responder a traves de este servidor, 435. El
trabajo no desaparece, se mueve del contexto facturado a la CPU, que no cuesta.

QUE DEVUELVE

Cada pasaje viene con su pseudopath -una ruta portable que empieza por @/ y se resuelve
contra el propio paquete-, con si el documento fue emitido o recibido, y con su relevancia.
El agente puede citar la fuente exacta en lugar de afirmar de memoria, que es la diferencia
entre una respuesta comprobable y uma plausible.

CONFIGURACION

El paquete y su clave se indican con variables de entorno, para que la clave no viaje en la
linea de comandos ni quede en el historial del intérprete:

    MDCX_ARCHIVO=/ruta/al/corpus.mdcx
    MDCX_CLAVE=la-clave-del-paquete

    python -m mdcx.servidor_mcp
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from . import formato

_ESTADO: dict = {}


def _conexion():
    """Abre el paquete una vez y lo reutiliza.

    Descifrar y descomprimir cuesta unas decimas de segundo, y no tiene sentido pagarlas en
    cada pregunta. La base vive en memoria: un paquete abierto no deja copia en claro en el
    disco que alguien pueda recoger despues.
    """
    if "con" in _ESTADO:
        return _ESTADO["con"]

    archivo = os.environ.get("MDCX_ARCHIVO", "").strip()
    clave = os.environ.get("MDCX_CLAVE", "")
    if not archivo:
        raise RuntimeError(
            "Falta MDCX_ARCHIVO: indique la ruta del paquete .mdcx a consultar.")
    ruta = Path(archivo)
    if not ruta.is_file():
        raise RuntimeError(f"No existe el paquete indicado: {ruta}")
    if not clave:
        raise RuntimeError("Falta MDCX_CLAVE: el paquete esta cifrado y no se puede abrir.")

    con, cabecera = formato.abrir(ruta, clave)
    _ESTADO["con"] = con
    _ESTADO["cabecera"] = cabecera
    return con


def crear_servidor():
    from mcp.server.mcpserver import MCPServer

    servidor = MCPServer(
        name="mdcx",
        title="Corpus documental consultable",
        instructions=(
            "Consulta un expediente documental ya convertido y verificado. Use buscar "
            "para localizar los pasajes que responden a una pregunta: devuelve el texto "
            "literal con la procedencia de cada uno, de modo que se pueda citar la fuente "
            "en lugar de afirmar de memoria. Use informacion para saber que contiene el "
            "corpus antes de preguntarle nada."
        ),
    )

    @servidor.tool(
        name="buscar",
        title="Buscar en el corpus",
        description=(
            "Busca en el expediente y devuelve los pasajes que responden a la consulta, "
            "cada uno con el documento del que sale, su ruta portable y si fue emitido o "
            "recibido. Acepta la pregunta en espanol aunque los documentos esten en ingles."
        ),
    )
    def buscar(consulta: str, maximo: int = 5,
               procedencia: str | None = None) -> dict:
        """Busca pasajes que respondan a la consulta.

        consulta: la pregunta, tal como se formularia a una persona.
        maximo: cuantos pasajes devolver, de 1 a 20.
        procedencia: "recibido" o "emitido" para restringir; omitir para buscar en todo.
        """
        con = _conexion()
        tope = max(1, min(int(maximo), 20))
        solo = procedencia.lower().strip() if procedencia else None
        if solo not in (None, "recibido", "emitido"):
            solo = None
        resultados = formato.consultar(con, consulta, maximo=tope, solo=solo)
        return {
            "consulta": consulta,
            "encontrados": len(resultados),
            "pasajes": [
                {
                    "documento": r["documento"],
                    "procedencia": r["origen"],
                    "ruta": r["pseudopath"],
                    "relevancia": r.get("score"),
                    "texto": r["parrafo"],
                }
                for r in resultados
            ],
        }

    @servidor.tool(
        name="informacion",
        title="Informacion del corpus",
        description=(
            "Describe que contiene el paquete: cuantos documentos, cuando se creo, quien lo "
            "emitio y con que fidelidad se convirtio desde los originales."
        ),
    )
    def informacion() -> dict:
        """Ficha del corpus abierto, sin consultarlo."""
        _conexion()
        cab = _ESTADO.get("cabecera", {})
        return {
            "formato": f"{cab.get('formato')} v{cab.get('version')}",
            "emisor": cab.get("emisor") or "(sin declarar)",
            "creado_utc": cab.get("creado_utc"),
            "documentos": cab.get("documentos"),
            "pasajes": cab.get("pasajes"),
            "integridad": "intacto" if cab.get("_integro") else "ALTERADO",
            "conversion": cab.get("conversion", {}),
        }

    @servidor.tool(
        name="documento",
        title="Leer un documento entero",
        description=(
            "Devuelve el texto completo de un documento del corpus, identificado por su "
            "nombre o por su ruta portable. Uselo solo cuando los pasajes no basten: un "
            "documento entero puede ocupar decenas de miles de tokens."
        ),
    )
    def documento(nombre: str) -> dict:
        """Texto integro de un documento del corpus."""
        con = _conexion()
        fila = con.execute(
            "SELECT d.nombre, d.pseudopath, d.origen, "
            "       group_concat(p.texto, char(10) || char(10)) "
            "FROM documento d JOIN pasaje p ON p.documento_id = d.id "
            "WHERE d.nombre = ? OR d.pseudopath = ? "
            "GROUP BY d.id ORDER BY p.orden LIMIT 1",
            (nombre, nombre)).fetchone()
        if not fila:
            return {"encontrado": False,
                    "aviso": f"No hay ningun documento llamado {nombre!r} en el corpus."}
        return {"encontrado": True, "documento": fila[0], "ruta": fila[1],
                "procedencia": fila[2], "texto": fila[3] or ""}

    return servidor


def main() -> int:
    # Se abre el paquete antes de arrancar, aunque las herramientas lo abririan solas al
    # primer uso. Un servidor mal configurado que arranca sin quejarse queda a la espera y
    # solo falla cuando un agente le pregunta algo, que es tarde y lejos del problema: quien
    # lo lanzo ya no esta mirando. Comprobarlo aqui convierte un fallo silencioso y diferido
    # en un mensaje inmediato con la causa.
    try:
        _conexion()
    except Exception as exc:  # noqa: BLE001
        print(f"No se pudo abrir el corpus: {exc}", file=sys.stderr)
        return 2

    try:
        servidor = crear_servidor()
    except Exception as exc:  # noqa: BLE001
        print(f"No se pudo iniciar el servidor: {exc}", file=sys.stderr)
        return 2

    cab = _ESTADO.get("cabecera", {})
    print(f"mdcx: {cab.get('documentos')} documentos listos para consultar.",
          file=sys.stderr)
    servidor.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
