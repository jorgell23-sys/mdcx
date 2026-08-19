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

"""mdcx: convierte un expediente documental a Markdown verificado y lo deja consultable.

El paquete resuelve tres cosas encadenadas:

  Convertir. Cada documento pasa a Markdown y se comprueba contra el texto que el original
  expone de verdad, con una libreria independiente del motor que convirtio. Lo que el motor
  estructurado no incluye se anexa literal en vez de darse por perdido.

  Empaquetar. El corpus, su indice de busqueda y la procedencia de cada pasaje caben en un
  solo archivo .mdcx, cifrado, que se puede mover y verificar sin abrirlo.

  Consultar. Una pregunta devuelve los pasajes que la responden con su fuente exacta, en
  milisegundos y sin pasar el expediente entero por el contexto de un modelo.
"""

__version__ = "1.0.0"

from . import buscar, formato  # noqa: F401

__all__ = ["buscar", "formato", "__version__"]
