import cv2
import numpy as np

OPTIONS = ("clahe", "sharpen", "deskew")


def read_image(path):
    # np.fromfile + imdecode also works for non-ASCII paths on Windows, unlike cv2.imread
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image {path}")
    return img


def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def apply_clahe(gray, clip=2.0, tile=8):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(gray)


def sharpen(gray, sigma=1.0, amount=0.5):
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)


def binarize(gray):
    """Otsu threshold, returned with text = 255."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # text is taken to be whichever side has fewer pixels, so light-on-dark signs get flipped
    if np.count_nonzero(bw) > bw.size / 2:
        bw = 255 - bw
    return bw


def dark_on_light(gray):
    """Models always see dark text on a light background."""
    # Decided from the border, not the pixel majority: a tight crop of a bold character can be
    # half ink, but its border is background. The segmenter keeps the majority rule in binarize.
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if border_value(gray) <= threshold:  # <= because Otsu returns 0 on a two-level image
        return 255 - gray
    return gray


def border_value(gray):
    edges = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    return int(np.median(edges))


def rotate(img, angle, fill):
    # canvas grows so the ends of a long word are not cut off
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w = int(round(w * cos + h * sin))
    new_h = int(round(h * cos + w * sin))
    m[0, 2] += (new_w - w) / 2
    m[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(img, m, (new_w, new_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


def estimate_skew(gray, max_angle=10, step=0.5):
    # the angle at which the row profile is most peaked (max sum of squares) is taken as level
    bw = binarize(gray)
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + step / 2, step):
        profile = rotate(bw, angle, 0).sum(axis=1, dtype=np.float64)
        score = float(np.sum(profile ** 2))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def deskew(gray, max_angle=10, min_angle=2):
    # the estimate is only good to about 2 degrees on short words, so smaller corrections are skipped
    angle = estimate_skew(gray, max_angle)
    if abs(angle) < min_angle:
        return gray
    return rotate(gray, angle, border_value(gray))


def preprocess(img, options=()):
    gray = to_gray(img)
    if "clahe" in options:
        gray = apply_clahe(gray)
    if "sharpen" in options:
        gray = sharpen(gray)
    if "deskew" in options:
        gray = deskew(gray)
    return gray


def resize(gray, w, h):
    shrinking = w * h < gray.shape[0] * gray.shape[1]
    return cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC)


def resize_to_height(gray, height=32):
    h, w = gray.shape
    return resize(gray, max(1, round(w * height / h)), height)


def pad_to_width(gray, width):
    h, w = gray.shape
    if w >= width:
        return resize(gray, width, h)
    return cv2.copyMakeBorder(gray, 0, 0, 0, width - w, cv2.BORDER_CONSTANT, value=border_value(gray))


def fit_square(gray, size=32, margin=2):
    """Scale a character crop to fit inside size x size, keep aspect ratio, centre it."""
    h, w = gray.shape
    scale = (size - 2 * margin) / max(h, w)
    small = resize(gray, max(1, round(w * scale)), max(1, round(h * scale)))
    sh, sw = small.shape
    top, left = (size - sh) // 2, (size - sw) // 2
    return cv2.copyMakeBorder(small, top, size - sh - top, left, size - sw - left,
                              cv2.BORDER_CONSTANT, value=border_value(gray))


def synthetic_word(text="Shop", dark_text=True):
    width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0][0] + 20
    img = np.full((60, width), 255 if dark_text else 30, np.uint8)
    cv2.putText(img, text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0 if dark_text else 230, 3)
    return img


if __name__ == "__main__":
    word = synthetic_word()
    dark, light = binarize(word), binarize(synthetic_word(dark_text=False))
    assert 0 < np.count_nonzero(dark) < dark.size / 2
    assert abs(np.count_nonzero(dark) - np.count_nonzero(light)) < 0.05 * np.count_nonzero(dark)

    # short mixed-case words (ascenders, descenders) can be off by ~2 degrees, so test on capitals
    tilted = rotate(synthetic_word("MEDICALS"), 6, 255)
    est = estimate_skew(tilted)
    assert abs(est + 6) <= 0.5, est
    assert abs(estimate_skew(word)) <= 0.5
    assert deskew(rotate(synthetic_word("MEDICALS"), 1, 255)).shape == rotate(synthetic_word("MEDICALS"), 1, 255).shape
    assert dark_on_light(synthetic_word(dark_text=False)).mean() > 128

    for opts in [(), ("clahe",), ("sharpen",), ("deskew",), OPTIONS]:
        out = preprocess(cv2.cvtColor(tilted, cv2.COLOR_GRAY2BGR), opts)
        assert out.ndim == 2 and out.dtype == np.uint8

    assert resize_to_height(word, 32).shape == (32, round(word.shape[1] * 32 / 60))
    assert pad_to_width(resize_to_height(word, 32), 200).shape == (32, 200)
    assert fit_square(word[5:55, 10:60], 32).shape == (32, 32)
    print("preprocess tests passed, estimated skew of a 6 degree tilt:", est)
