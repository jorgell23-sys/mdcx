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

"""Convert document collections to verified Markdown and make them queryable.

The package covers three stages:

Conversion
    Each document is converted to Markdown and checked against the text the
    original actually exposes, read with a library independent from the engine
    that performed the conversion. Content the structured engine omits is
    appended verbatim rather than reported as lost.

Packaging
    The resulting corpus, its search index and the provenance of every passage
    fit into a single encrypted ``.mdcx`` file. Its header can be read without
    the key, so the issuer and the integrity of a file can be verified before
    deciding to open it.

Retrieval
    A query returns the passages that answer it, each with its exact source, in
    milliseconds and without sending the whole collection through a model's
    context window.
"""

__version__ = "1.0.6"

from . import archive, search  # noqa: F401

__all__ = ["archive", "search", "__version__"]
