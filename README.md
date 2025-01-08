# kh_tex_sep
khmer text clusters with *correct* bbox 

![tldr](khmer_text_with_boxes.png)
tldr ^^^



usage example:

```python
from kh_bbox_gen import KhmerTextClusterGenerator

# define augmentations
augmentation_params = {
    'noise_factor': 0.08,
    'max_blur': 0.125,
    'use_background': True,
    'rotation_range': (-0.125, 0.125),
    'emboss': 0.0,
    'invert': 0.45
}
renderer = KhmerTextClusterGenerator(font_path="./Hanuman-Regular.ttf", font_size=24)
test_text = "សានកែវមនោរា"
clusters = renderer.render_text(test_text, "khmer_text_with_boxes.png")
```
