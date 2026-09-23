import numpy as np

from mayek.preprocess import load_gray, preprocess, size_features, stroke_width, to_dataset_scale


def test_shapes_and_ranges(make_glyph):
    img, skel, dist, meta = preprocess(make_glyph(0), 128)
    for a in (img, skel, dist):
        assert a.shape == (128, 128) and a.dtype == np.uint8
    assert img.max() == 255 and img[0, 0] == 0          # ink -> 1, paper -> 0
    assert (skel > 0).sum() < (img > 0).sum()          # the skeleton is thinner than the ink
    assert meta.tolist()[:2] == [24, 24]


def test_polarity_does_not_matter(make_glyph):
    g = make_glyph(1)
    a = preprocess(g, 64)
    b = preprocess(255 - g, 64)
    assert np.abs(a[0].astype(int) - b[0].astype(int)).max() <= 1


def test_crop_centres_the_character(make_glyph):
    g = np.full((40, 40), 240, np.uint8)
    g[2:14, 2:14] = make_glyph(2, size=12, thickness=1)  # small character in a corner
    img = preprocess(g, 64)[0]
    ys, xs = np.nonzero(img > 128)
    assert abs(ys.mean() - 31.5) < 6 and abs(xs.mean() - 31.5) < 6


def test_size_features():
    f = size_features(np.array([[24, 24, 12, 20, 0.2]], np.float32))
    assert f.shape == (1, 5) and np.isclose(f[0, 2], np.log(12))


def test_stroke_width_and_dataset_scale(make_glyph):
    big = make_glyph(0, size=400, thickness=6)
    assert 4 <= stroke_width(big < 128) <= 9
    small = to_dataset_scale(big, stroke_ratio=0.1)
    assert small.shape == (24, 24) and small.dtype == np.uint8
    assert stroke_width(small < 128) >= 1.5              # thin pen strokes were thickened
    scan = make_glyph(1)
    assert to_dataset_scale(scan) is not None and to_dataset_scale(scan).shape == scan.shape


def test_load_gray_rgba_is_white_paper():
    rgba = np.zeros((50, 50, 4), np.uint8)               # fully transparent canvas
    rgba[20:30, 20:30] = [0, 0, 0, 255]                  # one black stroke
    g = load_gray(rgba)
    assert g[0, 0] == 255 and g[25, 25] == 0
