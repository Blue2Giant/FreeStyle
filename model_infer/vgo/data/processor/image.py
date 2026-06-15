import random
from io import BytesIO

import cv2
import numpy as np
import torchvision
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter


def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def degrade_jpeg_compression(img, _random: random.Random):
    quality = _random.randint(5, 70)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)


def degrade_blur_defocus(img, _random: random.Random):
    """模拟模糊与失焦 (高斯、方框、运动模糊)"""
    blur_type = _random.choice(["gaussian", "box", "motion"])

    if blur_type == "gaussian":
        return img.filter(ImageFilter.GaussianBlur(radius=_random.uniform(1, 3)))
    elif blur_type == "box":
        return img.filter(ImageFilter.BoxBlur(_random.uniform(1, 2)))
    elif blur_type == "motion":
        cv_img = pil_to_cv2(img)
        ksize = _random.choice([5, 9, 15])
        # 生成水平运动模糊核
        kernel = np.zeros((ksize, ksize))
        kernel[int((ksize - 1) / 2), :] = np.ones(ksize)
        kernel = kernel / ksize
        cv_img = cv2.filter2D(cv_img, -1, kernel)
        return cv2_to_pil(cv_img)
    return img


def degrade_noise(img, _random: random.Random):
    cv_img = pil_to_cv2(img)
    noise = np.random.normal(0, _random.randint(5, 25), cv_img.shape).astype(np.float32)
    cv_img = np.clip(cv_img + noise, 0, 255).astype(np.uint8)
    return cv2_to_pil(cv_img)


def degrade_resolution(img, _random: random.Random):
    scale = _random.uniform(0.2, 0.7)
    w, h = img.size
    new_w, new_h = int(w * scale), int(h * scale)
    # 防止尺寸过小报错
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    return img_resized.resize((w, h), Image.Resampling.BILINEAR)


def degrade_occlusion(img, _random: random.Random):
    cv_img = pil_to_cv2(img)
    h, w, _ = cv_img.shape

    num_holes = _random.randint(1, 3)
    for _ in range(num_holes):
        x1 = _random.randint(0, w - 1)
        y1 = _random.randint(0, h - 1)

        # 限制遮挡块的大小
        w_occ = _random.randint(10, int(0.3 * w) if int(0.3 * w) > 10 else 11)
        h_occ = _random.randint(10, int(0.3 * h) if int(0.3 * h) > 10 else 11)

        x2 = min(w, x1 + w_occ)
        y2 = min(h, y1 + h_occ)

        # 用随机噪点填充遮挡区域
        cv_img[y1:y2, x1:x2] = np.random.randint(0, 255, (y2 - y1, x2 - x1, 3), dtype=np.uint8)

    return cv2_to_pil(cv_img)


def apply_random_degradations(img, min_ops=1, max_ops=4, _random: random.Random | None = None):
    if _random is None:
        _random = random

    assert _random is not None

    available_ops = [
        degrade_jpeg_compression,
        degrade_blur_defocus,
        degrade_noise,
        degrade_resolution,
    ]
    num_ops = _random.randint(min_ops, min(max_ops, len(available_ops)))

    selected_ops = _random.sample(available_ops, num_ops)

    for op in selected_ops:
        img = op(img, _random=_random)

    return img


def degrade_image(img: Image.Image, p=0.3, skip_rate=0.85, _random: random.Random | None = None) -> Image.Image:
    """Apply a sequence of randomized degradations to a PIL image.

    The function applies several independent degradation groups (each with
    probability `p`): JPEG compression, blur (gaussian/box/motion), additive
    noise, resolution down/up (simulated low-res), color/dynamic-range changes
    (brightness/contrast/saturation), and random occlusions/patches. Each group,
    when chosen, picks a random variant and strength.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image (expected RGB). A copy is operated on; the original is not
        modified.
    p : float, optional
        Per-degradation-group probability of applying that group (default 0.3).
        Each group is sampled independently.
    _random : random.Random | None, optional
        Optional Python random.Random instance to control Python-level randomness.
        If None, the global `random` module is used. Note: NumPy randomness
        (e.g., additive noise) uses `numpy.random` and is not controlled by
        `_random`; seed `numpy.random` separately for fully deterministic runs.

    Returns
    -------
    PIL.Image.Image
        Degraded PIL image.

    Notes
    -----
    - Motion blur is implemented via a simple horizontal kernel applied in OpenCV.
    - JPEG compression is performed by saving to an in-memory buffer with a
      random quality (5-70) and reopening the image.
    - Occlusions are random rectangular patches filled with random colors.
    - The function is intentionally simple and may use both Python and NumPy
      RNG sources; callers wanting full reproducibility should seed both.
    """
    if _random is None:
        _random = random

    assert _random is not None

    if _random.random() < skip_rate:
        return img

    img = img.copy()
    return apply_random_degradations(img, min_ops=1, max_ops=4, _random=_random)


def smart_resize(image, height: int, width: int):
    _, h, w = F.get_dimensions(image)

    image = F.resize(
        image,
        [height, width],
        torchvision.transforms.InterpolationMode.LANCZOS
        if height * width > h * w
        else torchvision.transforms.InterpolationMode.BILINEAR,
        antialias=True,
    )
    return image


def random_reference_pil_resize(image: Image.Image, *, height: int, width: int, _random: random.Random | None):
    if _random is None:
        _random = random
    assert _random is not None

    resize_mode = _random.choices(
        [
            Image.Resampling.BICUBIC,
            Image.Resampling.LANCZOS,
            Image.Resampling.BILINEAR,
            Image.Resampling.NEAREST,
            Image.Resampling.HAMMING,
            Image.Resampling.BOX,
        ],
        weights=[8, 2, 1, 1, 1, 1],  # HARD Code  PIL 随机 Resize 策略
        k=1,
    )[0]

    if resize_mode == Image.Resampling.BICUBIC:
        image = default_pil_resize(image, height=height, width=width)
    else:
        image = image.resize((width, height), resize_mode)

    return image


def default_pil_resize(image: Image.Image, *, height: int, width: int):
    """
    智能 Resize：根据缩放倍率决定是否开启 reducing_gap
    """
    orig_w, orig_h = image.size

    # 计算缩放倍数
    scale_w = orig_w / width
    scale_h = orig_h / height
    max_scale = max(scale_w, scale_h)

    # 阈值判断：
    # 如果原图是目标的 3 倍以上（例如 1500px -> 500px），开启 reducing_gap 防止混叠
    # 否则保持默认，以保留最大锐度和细节
    gap = 3.0 if max_scale > 3.0 else None

    return image.resize((width, height), resample=Image.Resampling.BICUBIC, reducing_gap=gap)


class AspectFit:
    def __init__(self, image_size: int, multi_size=False, size_base=32, range_scale=9 / 16, resolutions=None) -> None:
        if resolutions is None:
            self.resolutions = (
                self.generate_allowed_resolution(image_size, size_base, range_scale)
                if multi_size
                else np.array([[image_size, image_size]])
            )
        else:
            self.resolutions = np.array(resolutions)

        self.aspect_ratio = np.array([x[0] / x[1] for x in self.resolutions])

    @staticmethod
    def generate_allowed_resolution(image_size: int, size_base=32, range_scale=9 / 16):
        size_min = np.floor(np.sqrt(image_size * image_size * range_scale) / size_base).astype(np.int64) * size_base
        size_all = list(range(size_min, image_size, size_base))
        area = image_size * image_size
        aspect_size = []
        for size in size_all:
            if area % (size * size_base) == 0:
                aspect_size.append(
                    (
                        size,
                        np.ceil(area / size / size_base).astype(np.int64) * size_base,
                    )
                )
            else:
                aspect_size.append(
                    (
                        size,
                        np.ceil(area / size / size_base).astype(np.int64) * size_base,
                    )
                )
                aspect_size.append(
                    (
                        size,
                        np.floor(area / size / size_base).astype(np.int64) * size_base,
                    )
                )

        aspect_size = [*aspect_size, (image_size, image_size)]
        for width, height in aspect_size[::-1]:
            aspect_size.append((height, width))
        return np.array(sorted(set(aspect_size), key=lambda x: x[0] / x[1])).tolist()

    def match_by_wh(self, w, h):
        aspect_ratio_fact = w / h
        bucket_idx = np.argmin(np.abs(aspect_ratio_fact - self.aspect_ratio))
        return self.resolutions[bucket_idx]

    def __call__(self, image):
        _, h, w = F.get_dimensions(image)
        target_width, target_height = self.match_by_wh(w, h)

        scale = target_height / image.height if w / h > target_width / target_height else target_width / image.width

        width_scale = round(image.width * scale)
        height_scale = round(image.height * scale)

        image = smart_resize(image, height_scale, width_scale)

        delta_h = height_scale - target_height
        delta_w = width_scale - target_width

        # we assume that the image is already resized
        # such that the smallest size is at the desired size. Thus, eiter delta_h or delta_w must be zero
        assert delta_w >= 0 and delta_h >= 0 and not all([delta_h, delta_w])

        top = random.randint(0, delta_h)
        left = random.randint(0, delta_w)

        image = F.crop(image, top, left, target_height, target_width)  # type: ignore

        return image, (top, left)

    def __repr__(self):
        detail = f"(resolutions={self.resolutions})"
        return f"{self.__class__.__name__}{detail}"
