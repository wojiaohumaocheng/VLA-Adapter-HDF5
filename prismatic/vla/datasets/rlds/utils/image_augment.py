import numpy as np
import cv2
import copy
from typing import Dict, Union

def random_resized_crop_np(img, rng, scale, ratio):
    """
    img: float32, [H, W, C], 范围 [0,1]
    scale: [min_scale, max_scale]
    ratio: [min_ratio, max_ratio]
    """
    H, W, C = img.shape
    area = H * W

    scale_min, scale_max = scale
    ratio_min, ratio_max = ratio

    # 采样目标面积 & 宽高比
    target_area = area * rng.uniform(scale_min, scale_max)
    log_ratio = rng.uniform(np.log(ratio_min), np.log(ratio_max))
    r = np.exp(log_ratio)

    new_h = int(round(np.sqrt(target_area / r)))
    new_w = int(round(np.sqrt(target_area * r)))

    new_h = np.clip(new_h, 1, H)
    new_w = np.clip(new_w, 1, W)

    y = rng.integers(0, H - new_h + 1)
    x = rng.integers(0, W - new_w + 1)

    crop = img[y:y+new_h, x:x+new_w]
    # resize 回原大小
    crop_resized = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)
    return crop_resized


def random_brightness_np(img, rng, max_delta):
    # img: float32 [0,1]
    delta = rng.uniform(-max_delta, max_delta)
    img = img + delta
    return np.clip(img, 0.0, 1.0)


def random_contrast_np(img, rng, lower, upper):
    # img: float32 [0,1]
    factor = rng.uniform(lower, upper)
    mean = img.mean(axis=(0, 1), keepdims=True)
    img = (img - mean) * factor + mean
    return np.clip(img, 0.0, 1.0)


def random_saturation_np(img, rng, lower, upper):
    # RGB -> HSV 调整饱和度再转回去
    factor = rng.uniform(lower, upper)
    img_255 = (img * 255.0).astype(np.uint8)
    hsv = cv2.cvtColor(img_255, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    img_aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return img_aug.astype(np.float32) / 255.0


def random_hue_np(img, rng, max_delta):
    # hue 是 [0,180]，对应 TF 里 [-max_delta, max_delta] 的相对变化，这里大致来一个
    delta = rng.uniform(-max_delta, max_delta) * 180.0
    img_255 = (img * 255.0).astype(np.uint8)
    hsv = cv2.cvtColor(img_255, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + delta) % 180.0
    img_aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return img_aug.astype(np.float32) / 255.0


def augment_image_np(image: np.ndarray, rng: np.random.Generator, kwargs: dict) -> np.ndarray:
    """
    image: uint8, [H, W, 3]
    kwargs: 形如 image_augment_kwargs 那样的配置
    """
    # 转成 float32 [0,1]
    img = image.astype(np.float32) / 255.0

    order = kwargs.get("augment_order", [])

    for op in order:
        if op == "random_resized_crop" and "random_resized_crop" in kwargs:
            cfg = kwargs["random_resized_crop"]
            scale = cfg.get("scale", [1.0, 1.0])
            ratio = cfg.get("ratio", [1.0, 1.0])
            img = random_resized_crop_np(img, rng, scale, ratio)

        elif op == "random_brightness" and "random_brightness" in kwargs:
            max_delta = kwargs["random_brightness"][0]
            img = random_brightness_np(img, rng, max_delta)

        elif op == "random_contrast" and "random_contrast" in kwargs:
            lower, upper = kwargs["random_contrast"]
            img = random_contrast_np(img, rng, lower, upper)

        elif op == "random_saturation" and "random_saturation" in kwargs:
            lower, upper = kwargs["random_saturation"]
            img = random_saturation_np(img, rng, lower, upper)

        elif op == "random_hue" and "random_hue" in kwargs:
            max_delta = kwargs["random_hue"][0]
            img = random_hue_np(img, rng, max_delta)

    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)





def vmap_obs_np(fn, obs_seq: Dict) -> Dict:
    """
    obs_seq: dict of np.ndarray，所有数组形状 [T, ...]，T 是时间长度
    fn: 接受“单帧 obs dict”，返回“单帧 obs dict”
    """
    # 取一个数组看 T
    first_arr = next(iter(obs_seq.values()))
    first_arr = np.asarray(first_arr)
    T = first_arr.shape[0]

    collected = None

    for t in range(T):
        # 取出第 t 帧
        single = {k: np.asarray(v)[t] for k, v in obs_seq.items()}
        # 对单帧做增强
        single = fn(single)

        if collected is None:
            collected = {k: [] for k in single.keys()}

        for k, v in single.items():
            collected[k].append(v)

    # 把 list -> stack 回 [T, ...]
    out = {k: np.stack(v_list, axis=0) for k, v_list in collected.items()}
    return out


def apply_obs_transform_np(frame: Dict,
                           rng: np.random.Generator,
                           image_augment_kwargs: dict) -> Dict:
    """
    完整 NumPy 版：
    - 对 frame["task"] 做一次 augment_np
    - 对 frame["observation"] 在时间维上逐帧做 augment_np
    """
    frame = copy.deepcopy(frame)

    def fn_single(obs_single: Dict) -> Dict:
        return augment_np(obs_single, rng, image_augment_kwargs)

    # 1) task：一般是单帧 obs dict
    frame["task"] = fn_single(frame["task"])

    # 2) observation：dict of arrays，形状 [T, ...]
    frame["observation"] = vmap_obs_np(fn_single, frame["observation"])

    return frame

def augment_np(
    obs: Dict,
    rng: np.random.Generator,
    augment_kwargs: Union[Dict, Dict[str, Dict]],
) -> Dict:
    """
    NumPy 版 augment：
    - obs["image_xxx"] : (T, H, W, 3) 或 (H, W, 3) 的 uint8
    - obs["pad_mask_dict"]["image_xxx"] : (T,) 的 bool 数组
    只对 pad_mask 为 True 的帧做增强。
    """
    # 找出所有 image_xxx
    image_names = {key[6:] for key in obs if key.startswith("image_")}

    # 如果传进来的是一个总的 augment_kwargs（带 augment_order），复制给每个 image
    if "augment_order" in augment_kwargs:
        augment_kwargs = {name: augment_kwargs for name in image_names}

    pad_mask_dict = obs.get("pad_mask_dict", {})

    for i, name in enumerate(sorted(image_names)):
        if name not in augment_kwargs:
            continue
        kwargs = augment_kwargs[name]

        img_key = f"image_{name}"
        imgs = np.asarray(obs[img_key])          # (T, H, W, 3) 或 (H, W, 3)
        mask = pad_mask_dict.get(img_key, None)  # (T,) 或 None

        # ------- 单张图情况： (H, W, 3) -------
        if imgs.ndim == 3:
            # 没有 mask：默认做增强
            if mask is None:
                do_aug = True
            else:
                mask = np.asarray(mask)
                if mask.ndim == 0:
                    do_aug = bool(mask)
                else:
                    # 这里可以按需要选 any / all，这里用 any：只要有 True 就增强
                    do_aug = bool(mask.any())

            if do_aug:
                sub_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
                imgs = augment_image_np(imgs, sub_rng, kwargs)

            obs[img_key] = imgs
            continue

        # ------- 序列情况： (T, H, W, 3) -------
        if imgs.ndim == 4:
            T = imgs.shape[0]

            if mask is None:
                # 没有 mask：默认所有帧都做增强
                mask = np.ones(T, dtype=bool)
            else:
                mask = np.asarray(mask, dtype=bool)
                if mask.ndim == 0:
                    mask = np.full(T, bool(mask))
                elif mask.shape[0] != T:
                    raise ValueError(
                        f"pad_mask_dict[{img_key}] 长度 {mask.shape[0]} 和图像时间长度 {T} 不一致"
                    )

            # 对每一帧按 mask 决定是否增强
            for t in range(T):
                if mask[t]:
                    sub_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
                    imgs[t] = augment_image_np(imgs[t], sub_rng, kwargs)

            obs[img_key] = imgs
            continue

        # 其它维度不支持
        raise ValueError(f"不支持的图像维度 {imgs.ndim}，image_{name} 形状为 {imgs.shape}")

    return obs


def aug_np(frame: dict, image_augment_kwargs: dict) -> dict:
    rng = np.random.default_rng()   # 或者你自己管理 seed
    # task 一般是单帧；observation 是 (T, ...) 的序列
    frame["task"] = augment_np(frame["task"], rng, image_augment_kwargs)
    frame["observation"] = augment_np(frame["observation"], rng, image_augment_kwargs)
    return frame