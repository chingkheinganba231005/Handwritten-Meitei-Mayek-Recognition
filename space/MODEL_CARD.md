---
license: mit
library_name: timm
pipeline_tag: image-classification
tags:
  - handwriting-recognition
  - meitei-mayek
  - ocr
---

# Handwritten Meitei Mayek character recognition

Six networks fine-tuned on the 55-class TUMMHCD dataset of isolated
handwritten Meitei Mayek characters: ConvNeXt-T, EfficientNetV2-S and
ResNet-50-D, all ImageNet-pretrained, each trained once on the image alone and
once with five size features (the `_meta` files). Their average gets
{ensemble_accuracy}% top-1 accuracy on the 12,794 test images.

| File | What |
|---|---|
| `config.json` | members, test-time views, ensemble weights, normalisation statistics |
| `convnext_t.pt`, `effv2_s.pt`, `resnet50d_topo.pt` and their `_meta` versions (names as in `config.json`) | PyTorch state dicts |
| `web/model.onnx` | the EfficientNetV2-S member for the browser ({web_accuracy}% on its own), weights stored as float16 |

## Use

```bash
git clone {repo_url}
cd Handwritten-Meitei-Mayek-Recognition && pip install -r requirements.txt
huggingface-cli download {hf_id} --local-dir models
```

```python
from mayek.recognizer import Recognizer

rec = Recognizer("models")
preds, seen = rec.predict("photo.jpg", source="upload")   # or source="canvas" for drawings
print(preds[0].cls.display, preds[0].cls.name, preds[0].prob)
```

Inputs are brought to the scale of the dataset's 24 x 24 scans before the
usual preprocessing, so drawings and photos work, but accuracy on
handwriting unlike the TUMMHCD scans will be lower than the test figure.

Demo: [huggingface.co/spaces/{hf_id}](https://huggingface.co/spaces/{hf_id}).
Training code and every experiment: [{repo_url}]({repo_url}).

Dataset: D. Hijam and S. Saharia, "On developing complete character set
Meitei Mayek handwritten character database", *The Visual Computer* 38,
525–539 (2022).
