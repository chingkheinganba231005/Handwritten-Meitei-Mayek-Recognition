---
title: Handwritten Meitei Mayek Recognition
emoji: ✍️
colorFrom: blue
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Draw a Meitei Mayek character; a CNN reads it in your browser
models:
  - {hf_id}
---

# Handwritten Meitei Mayek recognition

Draw a single Meitei Mayek character, or upload a photo of one, and get the
top 5 predictions for all 55 classes of the TUMMHCD dataset: digits,
letters, final consonants, vowel signs and punctuation.

The page runs a fine-tuned EfficientNetV2-S ({web_accuracy}% top-1 on the
12,794 TUMMHCD test images) in your browser with ONNX Runtime Web. Nothing
you draw or upload is sent anywhere. The full three-network ensemble
({ensemble_accuracy}%) and all trained weights are in the model repository
[{hf_id}](https://huggingface.co/{hf_id}).

Code, experiments and limitations: [{repo_url}]({repo_url}).
