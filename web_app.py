"""Simple Flask web UI for handwriting synthesis."""

from __future__ import annotations

import argparse
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Tuple

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template_string,
    request,
    send_from_directory,
)

from demo import Hand


app = Flask(__name__)

DIST_DIR = Path(__file__).resolve().parent / "web_app" / "dist"
DIST_INDEX_FILE = DIST_DIR / "index.html"

TEMPLATE = """<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\">
    <title>Handwriting Synthesis</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f7f7f7; }
      header { background: #222; color: #fff; padding: 1.5rem; text-align: center; }
      main { max-width: 960px; margin: 2rem auto; background: #fff; padding: 2rem; box-shadow: 0 2px 6px rgba(0,0,0,.1); border-radius: 8px; }
      .workspace { display: flex; flex-direction: column; gap: 2rem; }
      @media (min-width: 900px) {
        .workspace { flex-direction: row; align-items: flex-start; }
        .preview { margin-top: 0; }
      }
      .input-column { flex: 1 1 50%; }
      form { width: 100%; }
      textarea { width: 100%; min-height: 240px; font-size: 1rem; padding: .75rem; border: 1px solid #ccc; border-radius: 4px; resize: vertical; }
      select { width: 100%; margin-top: .5rem; padding: .5rem; font-size: 1rem; border: 1px solid #ccc; border-radius: 4px; background: #fff; }
      button { margin-top: 1rem; background: #0b5ed7; color: #fff; border: none; padding: .75rem 1.5rem; font-size: 1rem; border-radius: 4px; cursor: pointer; }
      button:hover { background: #0a53be; }
      .error { margin-top: 1rem; color: #b00020; font-weight: bold; }
      p.help { color: #555; }
      .controls { display: grid; grid-template-columns: 1fr; gap: 1rem; margin-top: 1.5rem; }
      @media (min-width: 600px) {
        .controls { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      }
      .preview { flex: 1 1 50%; margin-top: 2rem; padding: 1rem; border: 1px solid #ddd; border-radius: 6px; background: #fafafa; min-height: 240px; display: flex; align-items: center; justify-content: center; }
      .preview.loading::after { content: "Rendering preview..."; color: #555; font-style: italic; }
      .preview svg { max-width: 100%; height: auto; }
    </style>
  </head>
  <body>
    <header>
      <h1>Handwriting Synthesis</h1>
      <p>Enter text below to generate an SVG rendition.</p>
    </header>
    <main>
      <div class=\"workspace\">
        <form method=\"post\" class=\"input-column\">
          <p class=\"help\">Each line may contain up to 75 characters and the supported character set is limited to A-Z, a-z, numbers, and basic punctuation.</p>
          <label for=\"text\" class=\"help\">Text to render</label>
          <textarea id=\"text\" name=\"text\" placeholder=\"Type or paste your message here...\" required>{{ text }}</textarea>
          <div class=\"controls\">
            <div>
              <label for=\"style\">Select a handwriting style</label>
              <select id=\"style\" name=\"style\">
              {% for style_id in styles %}
                <option value=\"{{ style_id }}\" {% if style_id == selected_style %}selected{% endif %}>Style {{ style_id }}</option>
              {% endfor %}
              </select>
            </div>
            <div>
              <label for=\"alignment\">Text alignment</label>
              <select id=\"alignment\" name=\"alignment\">
                <option value=\"left\" {% if selected_alignment == \"left\" %}selected{% endif %}>Left</option>
                <option value=\"center\" {% if selected_alignment == \"center\" %}selected{% endif %}>Center</option>
              </select>
            </div>
          </div>
          <button type=\"submit\">Generate SVG</button>
          {% if error %}<p class=\"error\">{{ error }}</p>{% endif %}
        </form>
        <section class=\"preview\" id=\"preview\">
          <p class=\"help\">Live preview will appear here as you type.</p>
        </section>
      </div>
    </main>
    <script>
      const textInput = document.getElementById('text');
      const styleSelect = document.getElementById('style');
      const preview = document.getElementById('preview');
      const alignmentSelect = document.getElementById('alignment');

      let debounceTimer;

      async function updatePreview() {
        const text = textInput.value;
        const style = styleSelect.value;
        const alignment = alignmentSelect.value;

        if (!text.trim()) {
          preview.classList.remove('loading');
          preview.innerHTML = '<p class="help">Live preview will appear here as you type.</p>';
          return;
        }

        preview.classList.add('loading');
        preview.innerHTML = '';

        try {
          const response = await fetch('/preview', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text, style, alignment })
          });

          if (!response.ok) {
            throw new Error('Request failed');
          }

          const data = await response.json();
          preview.classList.remove('loading');
          preview.innerHTML = data.svg;
        } catch (error) {
          preview.classList.remove('loading');
          preview.innerHTML = '<p class="error">Unable to render preview.</p>';
        }
      }

      function schedulePreview() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(updatePreview, 600);
      }

      textInput.addEventListener('input', schedulePreview);
      styleSelect.addEventListener('change', updatePreview);
      alignmentSelect.addEventListener('change', updatePreview);
      window.addEventListener('DOMContentLoaded', updatePreview);
    </script>
  </body>
</html>"""


@lru_cache(maxsize=1)
def get_hand() -> Hand:
    """Lazily create the Hand model instance."""

    return Hand()


def _available_styles() -> List[int]:
    """Discover the available handwriting styles from the styles directory."""

    style_ids = []
    styles_path = os.path.join(os.path.dirname(__file__), "styles")
    if not os.path.isdir(styles_path):
        return style_ids

    for filename in os.listdir(styles_path):
        if filename.startswith("style-") and filename.endswith("-strokes.npy"):
            try:
                style_id = int(filename.split("-")[1])
            except ValueError:
                continue
            style_ids.append(style_id)

    return sorted(set(style_ids))


def _prepare_generation_inputs(
    text_raw: str,
    style_raw,
    alignment_raw: str,
) -> Tuple[List[str], int | None, str]:
    """Validate and normalize inputs for preview/download requests."""

    if not isinstance(text_raw, str):
        raise ValueError("Please enter some text.")

    if not text_raw.strip():
        raise ValueError("Please enter some text.")

    lines = [line.rstrip() for line in text_raw.splitlines()]

    styles = _available_styles()
    default_style = styles[0] if styles else None

    try:
        style = int(style_raw) if style_raw is not None else default_style
    except (TypeError, ValueError):
        raise ValueError("Invalid style selected.") from None

    if style is not None and style not in styles:
        raise ValueError("Selected style is not available.")

    valid_alignments = {"left", "center"}
    alignment = alignment_raw if alignment_raw in valid_alignments else None
    if alignment is None:
        raise ValueError("Invalid alignment selected.")

    return lines, style, alignment


def _generate_svg(
    lines: Iterable[str], *, style: int | None = None, alignment: str = "center"
) -> bytes:
    """Generate SVG bytes for the provided lines."""

    if alignment not in {"left", "center"}:
        raise ValueError("Invalid alignment option.")

    hand = get_hand()

    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_file:
        temp_name = tmp_file.name

    try:
        kwargs = {"alignment": alignment}
        if style is not None:
            kwargs["styles"] = [style for _ in lines]

        hand.write(filename=temp_name, lines=lines, **kwargs)
        with open(temp_name, "rb") as fh:
            return fh.read()
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)


@app.route("/", methods=["GET", "POST"])
def index():
    """Render the form and handle SVG generation requests."""

    styles = _available_styles()
    default_style = styles[0] if styles else None
    valid_alignments = {"left", "center"}
    default_alignment = "center"

    if request.method == "GET":
        if DIST_INDEX_FILE.exists():
            return send_from_directory(DIST_DIR, "index.html")

        return render_template_string(
            TEMPLATE,
            error=None,
            text="",
            styles=styles,
            selected_style=default_style,
            selected_alignment=default_alignment,
        )

    if request.method == "POST":
        text = request.form.get("text", "")
        style_raw = request.form.get("style")
        alignment_raw = request.form.get("alignment", default_alignment)
        if alignment_raw not in valid_alignments:
            return render_template_string(
                TEMPLATE,
                error="Invalid alignment selected.",
                text=text,
                styles=styles,
                selected_style=default_style,
                selected_alignment=default_alignment,
            )

        alignment = alignment_raw
        if not text.strip():
            return render_template_string(
                TEMPLATE,
                error="Please enter some text.",
                text=text,
                styles=styles,
                selected_style=default_style,
                selected_alignment=alignment,
            )

        lines = [line.rstrip() for line in text.splitlines()]

        try:
            style = int(style_raw) if style_raw is not None else default_style
        except (TypeError, ValueError):
            return render_template_string(
                TEMPLATE,
                error="Invalid style selected.",
                text=text,
                styles=styles,
                selected_style=default_style,
                selected_alignment=alignment,
            )

        if style is not None and style not in styles:
            return render_template_string(
                TEMPLATE,
                error="Selected style is not available.",
                text=text,
                styles=styles,
                selected_style=default_style,
                selected_alignment=alignment,
            )

        try:
            svg_bytes = _generate_svg(lines, style=style, alignment=alignment)
        except ValueError as exc:
            return render_template_string(
                TEMPLATE,
                error=str(exc),
                text=text,
                styles=styles,
                selected_style=style,
                selected_alignment=alignment,
            )

        headers = {"Content-Disposition": "attachment; filename=handwriting.svg"}
        return Response(svg_bytes, mimetype="image/svg+xml", headers=headers)

    abort(405)


@app.route("/preview", methods=["POST"])
def preview() -> Response:
    """Return a JSON payload containing an inline SVG preview."""

    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    style_raw = payload.get("style")
    alignment_raw = payload.get("alignment", "center")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"svg": "<p class=\"help\">Enter text to see the preview.</p>", "error": "Please enter some text."})

    try:
        lines, style, alignment = _prepare_generation_inputs(text, style_raw, alignment_raw)
    except ValueError as exc:
        return jsonify({"svg": f"<p class=\"error\">{exc}</p>", "error": str(exc)}), 400

    try:
        svg_bytes = _generate_svg(lines, style=style, alignment=alignment)
    except ValueError as exc:
        return jsonify({"svg": f"<p class=\"error\">{exc}</p>", "error": str(exc)}), 400

    return jsonify({"svg": svg_bytes.decode("utf-8")})


@app.route("/api/styles", methods=["GET"])
def api_styles() -> Response:
    """Return the list of available handwriting styles."""

    styles = _available_styles()
    payload = {"styles": [{"id": style_id, "label": f"Style {style_id}"} for style_id in styles]}
    return jsonify(payload)


def _parse_generation_payload(require_text: bool = True) -> Tuple[List[str], int | None, str]:
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    style_raw = payload.get("style")
    alignment_raw = payload.get("alignment", "center")

    if not isinstance(text, str):
        raise ValueError("Please enter some text.")

    if require_text and not text.strip():
        raise ValueError("Please enter some text.")

    return _prepare_generation_inputs(text, style_raw, alignment_raw)


@app.route("/api/preview", methods=["POST"])
def api_preview() -> Response:
    """Generate an inline SVG preview for the React UI."""

    try:
        lines, style, alignment = _parse_generation_payload()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        svg_bytes = _generate_svg(lines, style=style, alignment=alignment)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({"svg": svg_bytes.decode("utf-8")})


@app.route("/api/generate", methods=["POST"])
def api_generate() -> Response:
    """Generate and return an SVG file for download."""

    try:
        lines, style, alignment = _parse_generation_payload()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        svg_bytes = _generate_svg(lines, style=style, alignment=alignment)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    headers = {"Content-Disposition": "attachment; filename=handwriting.svg"}
    return Response(svg_bytes, mimetype="image/svg+xml", headers=headers)


@app.route("/assets/<path:filename>")
def spa_assets(filename: str):
    """Serve compiled frontend asset files if they exist."""

    if not DIST_INDEX_FILE.exists():
        abort(404)

    assets_dir = DIST_DIR / "assets"
    return send_from_directory(assets_dir, filename)


@app.route("/<path:filename>")
def spa_catch_all(filename: str):
    """Serve the compiled SPA or fall back to legacy behaviour."""

    if filename.startswith("api/"):
        abort(404)

    if DIST_INDEX_FILE.exists():
        file_path = DIST_DIR / filename
        if file_path.is_file():
            return send_from_directory(DIST_DIR, filename)
        return send_from_directory(DIST_DIR, "index.html")

    abort(404)


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the web server entry point."""

    parser = argparse.ArgumentParser(description="Run the handwriting synthesis web UI.")
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind the development server to.")
    parser.add_argument(
        "--port",
        default=int(os.environ.get("PORT", 3000)),
        type=int,
        help="Port to listen on (defaults to 3000 or the PORT environment variable).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode (disabled by default to avoid the auto reloader).",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """Entry point for running the development server directly."""

    args = _parse_args(argv)
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=args.debug)


if __name__ == "__main__":
    main()

