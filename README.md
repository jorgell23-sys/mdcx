# mdcx

Convierte un expediente documental completo a Markdown comprobado, lo empaqueta en un solo
archivo cifrado, y lo deja consultable por agentes a traves del Model Context Protocol.

## El problema

Un agente que debe responder sobre un expediente tiene dos caminos. Puede recibir los
documentos enteros en su contexto, que es caro y ademas topa con el limite de la ventana. O
puede preguntar a algo que ya sepa donde esta cada cosa.

Medido sobre un expediente real de 99 documentos y 180 MB:

| | Tokens de modelo | Tokens locales |
|---|---|---|
| Leer los originales | **2.265.488** | 2.265.327 |
| Consultar el paquete | **435** | 2.688.861 |

El trabajo no desaparece: se mueve del contexto, que se factura y es finito, a la CPU, que
no cuesta. Al modelo solo llega el pasaje que responde.

## Las tres piezas

**Convertir.** Cada documento pasa a Markdown y se comprueba contra el texto que el original
expone de verdad, leido con una libreria independiente del motor que convirtio. Lo que el
motor estructurado no incluye se anexa literal en vez de darse por perdido. Sobre el
expediente de prueba: 99,948 % de cobertura, y 100 % en los documentos con texto.

**Empaquetar.** El corpus, su indice de busqueda y la procedencia de cada pasaje caben en un
`.mdcx`: un archivo cifrado con AES-256-GCM cuya cabecera se puede leer sin la clave, de
modo que se puede comprobar quien lo emitio y si esta integro antes de decidir abrirlo. De
8,7 MB de Markdown a 3,9 MB en un solo archivo.

**Consultar.** Una pregunta devuelve los pasajes que la responden con su fuente exacta. En
las 20 consultas reales del expediente, el documento correcto aparece entre los cinco
primeros resultados en 19 de 20, y entre los diez primeros en 20 de 20.

## Instalacion

```
pip install mdcx            # convertir y consultar
pip install "mdcx[mcp]"     # ademas, el servidor para agentes
pip install "mdcx[ocr]"     # ademas, reconocimiento optico para escaneados
```

## Uso

Convertir una carpeta, replicando su estructura:

```
mdcx-convertir --input ./Expediente --output ./Expediente_md
```

Empaquetar y consultar:

```
mdcx empaquetar --output ./Expediente_md --destino corpus.mdcx --clave "..."
mdcx info corpus.mdcx
mdcx buscar corpus.mdcx "donde se indica el diametro minimo a modelar" --clave "..."
mdcx exportar corpus.mdcx --destino ./recuperado --clave "..."
```

`exportar` esta por diseno: un formato del que no se puede salir es una trampa, por bien
intencionado que sea.

## Como servidor MCP

```json
{
  "mcpServers": {
    "mdcx": {
      "command": "python",
      "args": ["-m", "mdcx.servidor_mcp"],
      "env": {
        "MDCX_ARCHIVO": "/ruta/al/corpus.mdcx",
        "MDCX_CLAVE": "la-clave-del-paquete"
      }
    }
  }
}
```

Expone tres herramientas: `buscar` devuelve los pasajes que responden a una pregunta con su
procedencia; `informacion` describe que contiene el corpus y con que fidelidad se convirtio;
`documento` entrega un documento entero cuando los pasajes no bastan.

## Sobre las rutas

Ninguna salida contiene rutas absolutas. Cada documento se identifica por un *pseudopath*
que empieza por `@/` y se resuelve contra la carpeta o el paquete que lo contiene, de modo
que el corpus sigue siendo valido esté donde esté: disco local, red o nube.

## Sobre el cifrado

El paquete cifra en reposo y descifra en memoria al abrirlo; nada se escribe en claro en el
disco. Eso protege un archivo que circula. No es lo mismo que buscar sobre datos cifrados
sin descifrarlos nunca, que es un campo distinto, con ataques de fuga documentados y
sobrecostes de segundos por consulta.

La clave se deriva con scrypt, que hace lento probar claves: unos 8 intentos por segundo y
32 MB de memoria cada uno, lo que impide paralelizar en tarjeta grafica. Aun asi, **la
fortaleza real la decide la frase de paso**: una contrasena de diccionario cae en un dia.

## Autoria

Concebido y dirigido por **Jorge Ellena G.**, programado con la asistencia de Claude
(Anthropic).

Cada decision de este paquete se tomo contra mediciones sobre un expediente real, no por
costumbre: que motor de conversion usar, que licencia permite cual, como puntuar una
busqueda, que optimizaciones aceptar y cuales descartar. Varias se descartaron precisamente
por medirlas -reducir los candidatos de busqueda parecia acelerar diez veces y en realidad
hacia caer el acierto de 19 a 17 sobre 20-, y esas mediciones estan anotadas en el codigo
junto a la decision que justifican.

## Licencia

Apache 2.0. Se puede usar, modificar y vender, conservando el aviso de autoria.

Se evito deliberadamente PyMuPDF, cuya licencia AGPL obligaria a publicar bajo AGPL
cualquier programa que lo use, incluido el que solo lo ofrezca como servicio en red.
