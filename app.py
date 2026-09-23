"""Gradio demo: draw or upload one handwritten Meitei Mayek character.

    huggingface-cli download Chingkheinganba/handwritten-meitei-mayek-recognition --local-dir models
    python app.py

The public demo is the static page in web/, which runs in the browser.
"""

import json
import os
from pathlib import Path

import gradio as gr
import numpy as np

from mayek.preprocess import load_gray
from mayek.recognizer import Recognizer

MODEL_DIR = Path(os.environ.get("MAYEK_MODELS", "models"))
REPO_URL = "https://github.com/chingkheinganba231005/Handwritten-Meitei-Mayek-Recognition"

recognizer = Recognizer(MODEL_DIR)
config = json.loads((MODEL_DIR / "config.json").read_text())
MEMBERS = " + ".join({"convnext": "ConvNeXt-T", "effv2": "EfficientNetV2-S", "resnet50d": "ResNet-50-D"}.get(
    m["name"].split("_")[0], m["name"]) for m in config["members"])


def recognise(image, source, use_tta):
    if isinstance(image, dict):  # sketchpad value
        image = image.get("composite")
    if image is None or load_gray(image).min() > 245:
        return "Draw or upload a character first.", {}, None

    preds, seen = recognizer.predict(image, source=source, tta=use_tta, topk=5)
    top = preds[0].cls
    summary = f"# {top.display}\nclass {top.id} · {top.name} · {preds[0].prob:.1%}"
    if top.char is None:
        summary += "\n\nClass 054 has not been matched to a Unicode character yet."
    scores = {f"{p.cls.display}   {p.cls.id} · {p.cls.name}": p.prob for p in preds}
    return summary, scores, 255 - seen  # show dark ink on light paper


with gr.Blocks(title="Handwritten Meitei Mayek recognition") as demo:
    gr.Markdown(
        "## Handwritten Meitei Mayek recognition\n"
        "One character at a time, from all 55 classes of the TUMMHCD dataset: digits, letters, "
        "final consonants (lonsum), vowel signs and punctuation. Draw a single character on the "
        "canvas (pick the brush first), or upload a photo or scan of one character on light paper."
    )
    with gr.Row():
        with gr.Column():
            with gr.Tab("Draw"):
                sketch = gr.Sketchpad(
                    type="pil",
                    canvas_size=(400, 400),
                    brush=gr.Brush(default_size=16, colors=["#000000"], color_mode="fixed"),
                    label="Canvas",
                )
                draw_btn = gr.Button("Recognise drawing", variant="primary")
            with gr.Tab("Upload"):
                upload = gr.Image(type="pil", label="Character image")
                upload_btn = gr.Button("Recognise image", variant="primary")
            use_tta = gr.Checkbox(value=True, label="Test-time augmentation (averages 5 slightly shifted and zoomed views)")
        with gr.Column():
            summary = gr.Markdown()
            top5 = gr.Label(num_top_classes=5, label="Top 5")
            seen = gr.Image(label="What the networks see (128 x 128)", height=160, width=160, interactive=False)

    acc = config.get("test_accuracy")
    gr.Markdown(
        f"Ensemble of fine-tuned {MEMBERS}"
        + (f", {100 * acc:.2f}% top-1 accuracy on the 12,794 TUMMHCD test images" if acc else "")
        + ". The drawing is scaled down to the size of a dataset scan before recognition, so draw "
        f"the character large and fill the canvas as you would a writing box. Code: [GitHub]({REPO_URL})."
    )

    draw_btn.click(lambda im, t: recognise(im, "canvas", t), [sketch, use_tta], [summary, top5, seen])
    upload_btn.click(lambda im, t: recognise(im, "upload", t), [upload, use_tta], [summary, top5, seen])

if __name__ == "__main__":
    demo.launch()
