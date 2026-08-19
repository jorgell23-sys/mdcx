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

"""Conversion engines.

Docling understands layout, heading hierarchy and tables. Native extractors
do not interpret structure but are deterministic and depend on no model.

No engine uses a language model to transcribe: a generative model may
paraphrase, and the requirement is fidelity to the source.
"""
from __future__ import annotations

import os as _os
import html
import re
from pathlib import Path

_CONVERTERS: dict = {}

_DEVICE_CACHE: dict = {}

def gpu_available() -> bool:
    if "gpu" not in _DEVICE_CACHE:
        try:
            import torch

            _DEVICE_CACHE["gpu"] = bool(torch.cuda.is_available())
            _DEVICE_CACHE["name"] = (
                torch.cuda.get_device_name(0) if _DEVICE_CACHE["gpu"] else "CPU"
            )
        except Exception:
            _DEVICE_CACHE["gpu"] = False
            _DEVICE_CACHE["name"] = "CPU"
    return _DEVICE_CACHE["gpu"]

def device_name() -> str:
    gpu_available()
    return _DEVICE_CACHE.get("name", "CPU")

def _accelerator_options(threads: int = 4):
    """Configure Docling to use CUDA when available, falling back cleanly to CPU."""
    try:
        from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions

        device = AcceleratorDevice.CUDA if gpu_available() else AcceleratorDevice.AUTO
        return AcceleratorOptions(num_threads=threads, device=device)
    except Exception:
        return None

def local_artifacts_path():
    """Folder holding already downloaded models, if present."""
    from pathlib import Path as _P
    import os as _os

    candidatas = []
    env = _os.environ.get("DOCLING_ARTIFACTS_PATH")
    if env:
        candidatas.append(_P(env))
    candidatas.append(_P(__file__).resolve().parent.parent / "models")
    candidatas.append(_P.home() / ".cache" / "docling" / "models")
    for c in candidatas:
        if c.is_dir() and any(c.iterdir()):
            return c
    return None

def docling_available() -> bool:
    try:
        import docling  # noqa: F401

        return True
    except Exception:
        return False

def _get_docling_converter(ocr: bool):
    """Crea (y cachea) un DocumentConverter. Cachear importa: construirlo carga modelos."""
    key = f"ocr={ocr}"
    if key in _CONVERTERS:
        return _CONVERTERS[key]

    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    popts = PdfPipelineOptions()
    popts.do_ocr = ocr
    popts.do_table_structure = True
    accel = _accelerator_options()
    if accel is not None:
        popts.accelerator_options = accel

    artefactos = local_artifacts_path()
    if artefactos is not None:
        popts.artifacts_path = str(artefactos)
    try:
        from docling.datamodel.pipeline_options import TableFormerMode

        modo = _os.environ.get("PDFTOMD_TABLAS", "rapido").strip().lower()
        popts.table_structure_options.mode = (
            TableFormerMode.ACCURATE if modo in ("exacto", "accurate")
            else TableFormerMode.FAST)
        popts.table_structure_options.do_cell_matching = True
    except Exception:
        pass
    if ocr:
        try:
            from docling.datamodel.pipeline_options import RapidOcrOptions

            popts.ocr_options = RapidOcrOptions(
                lang=["english"],
                force_full_page_ocr=True,
                scale=4.0,
            )
        except Exception:
            try:
                popts.ocr_options.force_full_page_ocr = True
            except Exception:
                pass

    conv = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=popts)}
    )
    _CONVERTERS[key] = conv
    return conv

def _full_export_kwargs() -> dict:
    """Force Docling to export the whole document, not only the body."""
    kwargs: dict = {}
    try:
        from docling_core.types.doc.document import ContentLayer

        kwargs["included_content_layers"] = set(ContentLayer)
    except Exception:
        pass
    try:
        from docling_core.types.doc.labels import DocItemLabel

        kwargs["labels"] = set(DocItemLabel)
    except Exception:
        pass
    return kwargs

def docling_convert(path: Path, ocr: bool = False) -> tuple[str, dict, dict | None]:
    """Devuelve (markdown, metadatos, documento_json_sin_perdida)."""
    conv = _get_docling_converter(ocr)
    result = conv.convert(str(path))
    doc = result.document
    try:
        md = doc.export_to_markdown(**_full_export_kwargs())
    except Exception:
        md = doc.export_to_markdown()
    try:
        lossless = doc.export_to_dict()
    except Exception:
        lossless = None
    meta = {"engine": "docling", "ocr": ocr}
    try:
        meta["pages"] = len(doc.pages)
    except Exception:
        pass
    return md, meta, lossless

def native_pdf(path: Path) -> tuple[str, dict, None]:
    """PDF to Markdown by layout blocks, preserving reading order and page separation."""
    from . import pdf as _pdf

    doc = _pdf.open_document(path)
    out: list[str] = []
    try:
        for i, page in enumerate(doc, start=1):
            out.append(f"\n<!-- page {i} -->\n")
            for text in _pdf.page_paragraphs_fast(page):
                if text:
                    out.append(text)
        meta = {"engine": "pypdfium2", "pages": len(doc)}
    finally:
        doc.close()
    return "\n\n".join(out), meta, None

def _md_escape_cell(value) -> str:
    if value is None:
        return ""
    text = str(value).replace("|", "\\|")
    return " ".join(text.split())

def native_xlsx(path: Path) -> tuple[str, dict, None]:
    """XLSX to one Markdown table per sheet."""
    import openpyxl

    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    out: list[str] = []
    sheets = []
    try:
        for ws in wb.worksheets:
            sheets.append(ws.title)
            out.append(f"\n## Hoja: {ws.title}\n")
            rows = [list(r) for r in ws.iter_rows(values_only=True)]
            while rows and all(c is None for c in rows[-1]):
                rows.pop()
            if not rows:
                out.append("_(hoja vacia)_")
                continue
            width = max(len(r) for r in rows)
            header = [_md_escape_cell(c) for c in rows[0]] + [""] * (width - len(rows[0]))
            out.append("| " + " | ".join(header) + " |")
            out.append("|" + "---|" * width)
            for row in rows[1:]:
                cells = [_md_escape_cell(c) for c in row] + [""] * (width - len(row))
                out.append("| " + " | ".join(cells) + " |")
    finally:
        wb.close()
    return "\n".join(out), {"engine": "openpyxl", "sheets": sheets}, None

def native_docx(path: Path) -> tuple[str, dict, None]:
    """DOCX -> Markdown respetando niveles de title, listas y tables."""
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(str(path))
    out: list[str] = []

    def render_paragraph(p) -> str:
        text = p.text.strip()
        if not text:
            return ""
        style = (p.style.name or "").lower()
        if style.startswith("heading"):
            m = re.search(r"(\d+)", style)
            level = min(int(m.group(1)), 6) if m else 1
            return ("#" * level) + " " + text
        if "list" in style:
            return "- " + text
        return text

    def render_table(t) -> list[str]:
        rows = [
            [" ".join(c.text.split()).replace("|", "\\|") for c in r.cells] for r in t.rows
        ]
        if not rows:
            return []
        width = max(len(r) for r in rows)
        lines = [
            "",
            "| " + " | ".join(rows[0] + [""] * (width - len(rows[0]))) + " |",
            "|" + "---|" * width,
        ]
        for r in rows[1:]:
            lines.append("| " + " | ".join(r + [""] * (width - len(r))) + " |")
        lines.append("")
        return lines

    body = d.element.body
    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            line = render_paragraph(Paragraph(child, d))
            if line:
                out.append(line)
        elif tag == "tbl":
            out.extend(render_table(Table(child, d)))

    for section in d.sections:
        for name, container in (("Encabezado", section.header), ("Pie", section.footer)):
            texts = [p.text.strip() for p in container.paragraphs if p.text.strip()]
            if texts:
                out.append("\n<!-- " + name + ": " + " / ".join(texts) + " -->")

    return "\n\n".join(out), {"engine": "python-docx", "tables": len(d.tables)}, None

def native_text(path: Path) -> tuple[str, dict, None]:
    from .extract import _plain_text

    text, meta = _plain_text(path)
    meta["engine"] = "passthrough"
    return text, meta, None

def native_html(path: Path) -> tuple[str, dict, None]:
    raw = native_text(path)[0]
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>", "\n", raw)
    raw = re.sub(r"(?i)</(p|div|tr|h[1-6]|li)>", "\n\n", raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html.unescape(re.sub(r"[ \t]{2,}", " ", raw))
    return raw, {"engine": "html-strip"}, None

NATIVE = {
    "pdf": native_pdf,
    "xlsx": native_xlsx,
    "docx": native_docx,
    "pptx": None,
    "text": native_text,
    "html": native_html,
}
