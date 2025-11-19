"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""
import cv2
import os
import copy
from functools import partial
import fnmatch
import h5py
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type, Union, Callable
import numpy as np
import random
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from prismatic.models.backbones.llm.prompting import PromptBuilder, QwenPromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX, NUM_TOKENS
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.image_augment import *


@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False
    use_minivlm: bool = False


    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, current_action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"]

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")

        # Get future action chunk
        future_actions = rlds_batch["action"][1:]

        if self.use_minivlm:
            self.prompt_builder_fn = QwenPromptBuilder
            prompt_builder = self.prompt_builder_fn("openvla")
            # Get action chunk string
            future_actions_string = self.action_tokenizer(future_actions,self.use_minivlm)
            current_action_string = self.action_tokenizer(current_action,self.use_minivlm)

            action_chunk_string = [current_action_string] + future_actions_string
            flattened_action_chunk_string = [item for sublist in action_chunk_string for item in sublist]
            action_chunk_len = len(flattened_action_chunk_string) 

            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": ''},
            ]

            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])

            prompt = prompt_builder.get_prompt() #e.g. 'In: What action should the robot take to put both the cream cheese box and the butter in the basket?\nOut: 希</s>'
            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids

            if len(input_ids) >= 3:
                del input_ids[-3] 
                del input_ids[-2] 
                del input_ids[-1] 

            if NUM_TOKENS<len(flattened_action_chunk_string):
                input_ids = input_ids + flattened_action_chunk_string[:NUM_TOKENS]
            else:
                remaining_length = NUM_TOKENS - len(flattened_action_chunk_string)
                extended_array = random.choices(flattened_action_chunk_string, k=remaining_length)
                
                input_ids = input_ids + flattened_action_chunk_string + extended_array
            labels = list(input_ids)
            action_chunk_len = NUM_TOKENS

        else:
            future_actions_string = ''.join(self.action_tokenizer(future_actions, use_minivlm=False))

            # Get action chunk string
            current_action_string = self.action_tokenizer(current_action, use_minivlm=False)
            action_chunk_string = current_action_string + future_actions_string
            action_chunk_len = len(action_chunk_string)

            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": action_chunk_string[0]},
            ]
            # remove action token
            # conversation = [
            #     {"from": "human", "value": f"What action should the robot take to {lang}?"},
            #     {"from": "gpt", "value": ""},
            # ]
            action_chunk_len = 1


            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])
            prompt = prompt_builder.get_prompt() #e.g. 'In: What action should the robot take to put both the cream cheese box and the butter in the basket?\nOut: 希</s>'
            # Tokenize (w/ `base_tokenizer`)
            input_ids = self.base_tokenizer(prompt, add_special_tokens=True).input_ids
            labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return_dict = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=actions)

        # Add additional inputs
        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            return_dict["proprio"] = proprio

        return return_dict
    
@dataclass
class HDF5BatchTransform:
    """
    专门给 HDF5Dataset 用的 BatchTransform：
    - 输入：一条轨迹里采出来的 K 个时间步（HDF5Dataset 已经做好 uniform / chunk / resize / 采样）
    - 输出：各时间步的 token、图像、动作等，带一个时间维 K，交给 collator 再展平。
    """
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False
    use_minivlm: bool = False

    def _get_lang_str(self, language_instruction: Any) -> str:
        """把各种奇怪类型的 language_instruction 统一成小写 str。"""
        if isinstance(language_instruction, (np.ndarray, list)):
            language_instruction = language_instruction[0]
        if isinstance(language_instruction, (np.str_, str)):
            return language_instruction.lower()
        if isinstance(language_instruction, (np.bytes_, bytes)):
            return language_instruction.decode().lower()
        return str(language_instruction).lower()

    def __call__(self, hdf5_batch: Dict[str, Any]) -> Dict[str, Any]:
        actions = np.asarray(hdf5_batch["action"])      # [K, NUM_ACTIONS_CHUNK, ACTION_DIM]
        K = actions.shape[0]

        # -------- 1. 语言指令（整条轨迹一般都是同一句，只取第一个就行） --------
        lang = self._get_lang_str(hdf5_batch["task"]["language_instruction"])

        dataset_name = hdf5_batch["dataset_name"]  # 可以是 [K] 或标量，保持原样，留给 collator 展平

        # -------- 2. 图像：primary 相机，保持 K 维度 --------
        primary_imgs_np = hdf5_batch["observation"]["image_primary"]  # [K, H, W, C]
        primary_imgs_pil = [Image.fromarray(primary_imgs_np[i]) for i in range(K)]
        pv_list = [self.image_transform(img) for img in primary_imgs_pil]  # 每个 [C, H, W]
        pixel_values = torch.stack(pv_list, dim=0)  # [K, C, H, W]

        # -------- 3. wrist 图（如果用的话），同样保持 K 维度 --------
        pixel_values_wrist = None
        if self.use_wrist_image:
            wrist_keys = [k for k in hdf5_batch["observation"].keys() if "wrist" in k]
            if len(wrist_keys) > 0:
                wrist_list = []
                for i in range(K):
                    per_step_wrist = []
                    for k in wrist_keys:
                        img_wrist = Image.fromarray(hdf5_batch["observation"][k][i])
                        pv_wrist = self.image_transform(img_wrist)  # [C, H, W]
                        per_step_wrist.append(pv_wrist)
                    # 和你原来一样，沿着 C 维拼接
                    per_step_wrist = torch.cat(per_step_wrist, dim=0)  # [C_total, H, W]
                    wrist_list.append(per_step_wrist)
                pixel_values_wrist = torch.stack(wrist_list, dim=0)  # [K, C_total, H, W]

        # -------- 4. proprio（如果用的话），保持 [K, PROPRIO_DIM] --------
        proprio = None
        if self.use_proprio and "proprio" in hdf5_batch["observation"]:
            proprio = np.asarray(hdf5_batch["observation"]["proprio"], dtype=np.float32)  # [K, D]

        # -------- 5. 为 K 个时间步分别构造 prompt / input_ids / labels --------
        input_ids_list = []
        labels_list = []

        # 非 minivlm 分支里，action_chunk_len 恒为 1；minivlm 分支恒为 NUM_TOKENS
        # 所以不用单独存 per-step 的长度
        for i in range(K):
            current_action = actions[i, 0]     # [ACTION_DIM]
            future_actions = actions[i, 1:]    # [NUM_ACTIONS_CHUNK-1, ACTION_DIM]

            # 构造 prompt_builder
            if self.use_minivlm:
                # 和你原来一样，minivlm 用 QwenPromptBuilder
                pb_cls = QwenPromptBuilder
            else:
                pb_cls = self.prompt_builder_fn
            prompt_builder = pb_cls("openvla")

            if self.use_minivlm:
                # ----- minivlm 分支：把动作token追加到 input_ids 结尾 -----
                future_actions_string = self.action_tokenizer(future_actions, self.use_minivlm)
                current_action_string = self.action_tokenizer(current_action, self.use_minivlm)

                action_chunk_string = [current_action_string] + future_actions_string
                flattened_action_chunk_string = [
                    token for sublist in action_chunk_string for token in sublist
                ]

                conversation = [
                    {"from": "human", "value": f"What action should the robot take to {lang}?"},
                    {"from": "gpt",   "value": ''},
                ]
                for turn in conversation:
                    prompt_builder.add_turn(turn["from"], turn["value"])

                prompt = prompt_builder.get_prompt()
                input_ids = self.base_tokenizer(prompt, add_special_tokens=True).input_ids

                # 你原来为了去掉结尾的特殊 token 的那三步，这里也保留
                if len(input_ids) >= 3:
                    del input_ids[-3]
                    del input_ids[-2]
                    del input_ids[-1]

                # 动作 token 固定填满 NUM_TOKENS
                if NUM_TOKENS < len(flattened_action_chunk_string):
                    action_tokens = flattened_action_chunk_string[:NUM_TOKENS]
                else:
                    remaining = NUM_TOKENS - len(flattened_action_chunk_string)
                    extended = random.choices(flattened_action_chunk_string, k=remaining)
                    action_tokens = flattened_action_chunk_string + extended

                input_ids = input_ids + action_tokens
                labels = list(input_ids)
                action_chunk_len = NUM_TOKENS

            else:
                # ----- 非 minivlm 分支：只用第一个 action token 作为输出 -----
                future_actions_string = ''.join(
                    self.action_tokenizer(future_actions, use_minivlm=False)
                )
                current_action_string = self.action_tokenizer(
                    current_action, use_minivlm=False
                )
                action_chunk_string = current_action_string + future_actions_string

                # 只用第一个动作 token 作为 gpt 首 token
                conversation = [
                    {"from": "human", "value": f"What action should the robot take to {lang}?"},
                    {"from": "gpt",   "value": action_chunk_string[0]},
                ]
                action_chunk_len = 1

                for turn in conversation:
                    prompt_builder.add_turn(turn["from"], turn["value"])

                prompt = prompt_builder.get_prompt()
                input_ids = self.base_tokenizer(prompt, add_special_tokens=True).input_ids
                labels = list(input_ids)

            # 转成 tensor，并把非动作部分 label 置为 IGNORE_INDEX
            input_ids_t = torch.tensor(input_ids, dtype=torch.long)
            labels_t = torch.tensor(labels, dtype=torch.long)

            labels_t[: -(action_chunk_len + 1)] = IGNORE_INDEX
            if not self.predict_stop_token:
                labels_t[-1] = IGNORE_INDEX

            input_ids_list.append(input_ids_t)
            labels_list.append(labels_t)

        # -------- 6. 把 K 个时间步的 input_ids / labels pad 成 [K, L_max] --------
        pad_token_id = (
            self.base_tokenizer.pad_token_id
            if self.base_tokenizer.pad_token_id is not None
            else 0
        )
        max_len = max(x.size(0) for x in input_ids_list)

        padded_inputs = []
        padded_labels = []
        for ids_t, lab_t in zip(input_ids_list, labels_list):
            pad_len = max_len - ids_t.size(0)
            if pad_len > 0:
                ids_pad = torch.full(
                    (pad_len,), pad_token_id, dtype=torch.long
                )
                lab_pad = torch.full(
                    (pad_len,), IGNORE_INDEX, dtype=torch.long
                )
                ids_t = torch.cat([ids_t, ids_pad], dim=0)
                lab_t = torch.cat([lab_t, lab_pad], dim=0)
            padded_inputs.append(ids_t)
            padded_labels.append(lab_t)

        input_ids = torch.stack(padded_inputs, dim=0)  # [K, L_max]
        labels = torch.stack(padded_labels, dim=0)     # [K, L_max]

        # -------- 7. 组装返回字典，保留 K 这条时间维 --------
        return_dict: Dict[str, Any] = dict(
            pixel_values=pixel_values,     # [K, C, H, W]
            input_ids=input_ids,           # [K, L_max]
            labels=labels,                 # [K, L_max]
            dataset_name=dataset_name,     # [K] 或标量，交给 collator 展平
            actions=actions,               # [K, NUM_ACTIONS_CHUNK, ACTION_DIM]
        )

        if pixel_values_wrist is not None:
            return_dict["pixel_values_wrist"] = pixel_values_wrist  # [K, C_total, H, W]
        if proprio is not None:
            return_dict["proprio"] = proprio                        # [K, PROPRIO_DIM]

        return return_dict


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        if "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=NUM_ACTIONS_CHUNK-1,      # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)


class HDF5Dataset(Dataset):
    def __init__(
            self,
            data_dir,
            dataset_name,
            image_augment_kwargs=None,
            batch_transform=None,
            resize_resolution=(224, 224),
            depth_resize_resolution=(224, 224),
            train=True,
            image_aug=True,
            num_samples_per_traj: int = 4,
    ):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.file_paths = self.traverse_hdf5_file()
        self.dataset_statistics = self.get_dataset_statistics()
        self.train_mode = train
        self.resize_resolution = resize_resolution
        self.depth_resize_resolution = depth_resize_resolution
        self.image_augment_kwargs = image_augment_kwargs
        self.image_aug = image_aug
        self.batch_transform = batch_transform
        self.num_samples_per_traj = num_samples_per_traj


    def traverse_hdf5_file(self):
        file_paths = []
        for root, _, files in os.walk(self.data_dir):
            for filename in fnmatch.filter(files, '*.hdf5'):
                file_path = os.path.join(root, filename)
                file_paths.append(file_path)
        return file_paths

    @staticmethod
    def read_one_file(path, read_arrays=True):
        # 若你的数据文件可能被同时写入，读时可用 swmr=True（前提：文件以 SWMR 写出）
        try:
            with h5py.File(path, 'r') as f:
                act_ds = f['action']
                qpos_ds = f['observations']['qpos']
                n = act_ds.shape[0]
                if read_arrays:
                    # 读入内存；如太大可按需切片/分块
                    actions = act_ds[...]
                    qpos = qpos_ds[...]
                    return {'num_transitions': n, 'actions': actions, 'qpos': qpos}
                else:
                    # 只做统计，避免内存压力
                    return {
                        'num_transitions': n,
                        'action_shape': act_ds.shape,
                        'qpos_shape': qpos_ds.shape,
                        'dtype_action': act_ds.dtype.str,
                        'dtype_qpos': qpos_ds.dtype.str,
                    }
        except Exception as e:
            print("{},{}".format(e, path))





    def parallel_load(self, file_paths, read_arrays=True, max_workers=1):
        actions, proprios = [], []
        num_transitions = 0
        max_workers = max(len(file_paths) if os.cpu_count() > len(file_paths) else os.cpu_count(), max_workers )
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(self.read_one_file, p, read_arrays) for p in file_paths]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="calculating dataset stats"):
                res = fut.result()
                num_transitions += res['num_transitions']
                if read_arrays:
                    actions.append(res['actions'])
                    proprios.append(res['qpos'])
        return actions, proprios, num_transitions


    def get_dataset_statistics(self,):
        logging.info("calculating dataset statistics...")
        actions, proprios, num_transitions = self.parallel_load(self.file_paths)
        num_trajectories = len(self.file_paths)
        actions, proprios = np.concatenate(actions), np.concatenate(proprios)
        metadata = {
            "action": {
                "mean": actions.mean(0).tolist(),
                "std": actions.std(0).tolist(),
                "max": actions.max(0).tolist(),
                "min": actions.min(0).tolist(),
                "q01": np.quantile(actions, 0.01, axis=0).tolist(),
                "q99": np.quantile(actions, 0.99, axis=0).tolist(),
            },
            "proprio": {
                "mean": proprios.mean(0).tolist(),
                "std": proprios.std(0).tolist(),
                "max": proprios.max(0).tolist(),
                "min": proprios.min(0).tolist(),
                "q01": np.quantile(proprios, 0.01, axis=0).tolist(),
                "q99": np.quantile(proprios, 0.99, axis=0).tolist(),
            },
            "num_transitions": num_transitions,
            "num_trajectories": num_trajectories,
        }
        dataset_statistics = {self.dataset_name: metadata}

        return dataset_statistics

    def tree_merge(self, *trees: Dict) -> Dict:
        merged = {}
        for tree in trees:
            for k, v in tree.items():
                if isinstance(v, dict):
                    merged[k] = self.tree_merge(merged.get(k, {}), v)
                else:
                    merged[k] = v
        return merged

    def get_first_leaf(self, struct):
        """从嵌套 dict 里拿到第一个非 dict 的叶子"""
        if isinstance(struct, dict):
            for v in struct.values():
                return self.get_first_leaf(v)
            raise ValueError("Empty dict in observation.")
        else:
            return struct

    def gather_tree(self, struct, idxs):
        """
        对嵌套 dict 里的所有叶子做索引：
        叶子假设是形状 [T, ...] 的 np.ndarray
        """
        if isinstance(struct, dict):
            return {k: self.gather_tree(v, idxs) for k, v in struct.items()}
        else:
            arr = np.asarray(struct)
            return arr[idxs]

    def uniform_np(self, traj:Dict) -> Dict:
        """用 NumPy 实现的 uniform relabel."""
        # 1. 先拿到一个叶子，确定轨迹长度 T
        first_leaf = self.get_first_leaf(traj["observation"])
        first_leaf = np.asarray(first_leaf)
        traj_len = first_leaf.shape[0]  # T

        # 2. 为每个时间步 i 采一个随机的未来 index
        goal_idxs = np.empty(traj_len, dtype=np.int32)
        for i in range(traj_len - 1):
            goal_idxs[i] = np.random.randint(i + 1, traj_len)  # [i+1, traj_len)
        # 最后一个时间步没有未来，索引自己
        goal_idxs[traj_len - 1] = traj_len - 1

        # 3. 对 observation 整棵树做 gather
        goal = self.gather_tree(traj["observation"], goal_idxs)

        # 4. 合并到 traj["task"] 里
        traj["task"] = self.tree_merge(traj.get("task", {}), goal)
        return traj


    def chunk_act_obs_np(self, traj: Dict, window_size: int, future_action_window_size: int = 0) -> Dict:
        """
        NumPy 实现的 chunk_act_obs：
        - obs: [T, ...] -> [T - future_action_window_size, window_size, ...]
        - act: [T, A]   -> [T - future_action_window_size, window_size + future_action_window_size, A]
        """
        actions = np.asarray(traj["action"])
        traj_len = actions.shape[0]  # T
        effective_traj_len = traj_len - future_action_window_size
        if effective_traj_len <= 0:
            raise ValueError("trajectory too short for given future_action_window_size")

        # ------- 构造 observation 的窗口索引 [effective_traj_len, window_size] -------
        # 每一行是: i-window+1 ... i-1, i
        base = np.arange(-window_size + 1, 1)  # [-w+1, ..., 0]  shape: [window_size]
        offsets = np.arange(effective_traj_len)[:, None]  # [[0], [1], ..., [E-1]] shape: [E,1]
        chunk_indices = offsets + base  # shape: [E, window_size]

        # 小于 0 的用 0 填（复制第 0 帧）
        floored_chunk_indices = np.maximum(chunk_indices, 0)

        # ------- 构造 action 的窗口索引 [effective_traj_len, window_size + future_action_window_size] -------
        base_a = np.arange(-window_size + 1, 1 + future_action_window_size)
        action_chunk_indices = offsets + base_a  # shape: [E, window_size + future]
        # 两端都裁剪到 [0, traj_len - 1]
        floored_action_chunk_indices = np.clip(action_chunk_indices, 0, traj_len - 1)

        # ------- 对 observation 整棵树做 gather，新增一个窗口维度 -------
        traj["observation"] = self.gather_tree(traj["observation"], floored_chunk_indices)
        for k, v in traj["observation"].items():
            if isinstance(v, np.ndarray) and v.ndim >= 2:
                traj["observation"][k] = np.squeeze(v, axis=1)

        # ------- 对 action 做 gather -------
        traj["action"] = actions[floored_action_chunk_indices]

        # ------- pad_mask：True 表示真实（index >= 0），False 表示左侧 padding -------
        traj["observation"]["pad_mask"] = (chunk_indices >= 0)

        # ------- 截断其他字段到有效长度 -------
        idxs_short = np.arange(effective_traj_len)

        if "task" in traj:
            traj["task"] = self.gather_tree(traj["task"], idxs_short)

        if "dataset_name" in traj:
            traj["dataset_name"] = np.asarray(traj["dataset_name"])[idxs_short]

        if "absolute_action_mask" in traj:
            traj["absolute_action_mask"] = np.asarray(traj["absolute_action_mask"])[idxs_short]

        return traj

    def resize_image_batched(self, images: np.ndarray, size):
        """
        images: [..., H, W, C]，最后三维是图像
        size: (new_h, new_w)
        """
        images = np.asarray(images)
        *prefix, H, W, C = images.shape
        new_h, new_w = size

        flat = images.reshape(-1, H, W, C)  # [N, H, W, C]
        resized = []
        for img in flat:
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            resized.append(img_resized)
        resized = np.stack(resized, axis=0)  # [N, new_h, new_w, C]
        return resized.reshape(*prefix, new_h, new_w, C)

    def resize_depth_batched(self, depths: np.ndarray, size):
        """
        depths: [..., H, W] 或 [..., H, W, 1]
        size: (new_h, new_w)
        """
        depths = np.asarray(depths, dtype=np.float32)
        orig_ndim = depths.ndim
        if orig_ndim == 3:
            # [..., H, W] => [..., H, W, 1]
            depths = depths[..., None]

        *prefix, H, W, C = depths.shape  # C 应该为 1
        new_h, new_w = size

        flat = depths.reshape(-1, H, W, C)
        resized = []
        for d in flat:
            d_resized = cv2.resize(d, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            resized.append(d_resized)
        resized = np.stack(resized, axis=0).reshape(*prefix, new_h, new_w, C)

        if orig_ndim == 3:
            resized = resized[..., 0]  # 再挤掉通道维

        return resized

    def decode_and_resize_np(
            self,
            obs: Dict,
            resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]],
            depth_resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]],
    ) -> Dict:
        """
        NumPy 版：
        obs 里可以是：
          image_xxx: [..., H, W, 3] 或空/None
          depth_xxx: [..., H, W] 或 [..., H, W, 1] 或空/None
        前面所有维度都当 batch 维处理（比如 T 或 B,T）
        """
        image_names = {key[6:] for key in obs if key.startswith("image_")}
        depth_names = {key[6:] for key in obs if key.startswith("depth_")}

        # 统一 resize 大小：tuple => 所有 image 共用
        if isinstance(resize_size, tuple):
            resize_size = {name: resize_size for name in image_names}
        if isinstance(depth_resize_size, tuple):
            depth_resize_size = {name: depth_resize_size for name in depth_names}

        # ---------- 处理 RGB 图 ----------
        for name in image_names:
            if name not in resize_size:
                logging.warning(
                    f"No resize_size was provided for image_{name}. This will result in 1x1 "
                    "padding images, which may cause errors if you mix padding and non-padding images."
                )

            key = f"image_{name}"
            image = obs[key]

            # padding 情况：None 或 空数组
            if image is None or (isinstance(image, np.ndarray) and image.size == 0):
                h, w = resize_size.get(name, (1, 1))
                # 这里默认只有单帧 padding，如果你有 batch 维也想 padding，可以根据需要扩展
                image = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                image = np.asarray(image)
                if image.dtype != np.uint8:
                    raise ValueError(f"Unsupported image dtype: found {key} with dtype {image.dtype}")

                if name in resize_size:
                    image = self.resize_image_batched(image, size=resize_size[name])

            obs[key] = image

        # ---------- 处理 depth 图 ----------
        for name in depth_names:
            if name not in depth_resize_size:
                logging.warning(
                    f"No depth_resize_size was provided for depth_{name}. This will result in 1x1 "
                    "padding depth images, which may cause errors if you mix padding and non-padding images."
                )

            key = f"depth_{name}"
            depth = obs[key]

            if depth is None or (isinstance(depth, np.ndarray) and depth.size == 0):
                h, w = depth_resize_size.get(name, (1, 1))
                depth = np.zeros((h, w, 1), dtype=np.float32)
            else:
                depth = np.asarray(depth, dtype=np.float32)

                if name in depth_resize_size:
                    depth = self.resize_depth_batched(depth, size=depth_resize_size[name])
            obs[key] = depth

        return obs

    from functools import partial

    def apply_obs_transform_np(
            self,
            frame: Dict,
            resize_size,
            depth_resize_size
    ) -> Dict:
        """
        完整 NumPy 版：
        - frame["task"]：通常是“单个 obs dict”（没有时间维），fn 会处理其中所有 image_*/depth_*
        - frame["observation"]：dict of arrays，可能是 [T,...] 或 [B,T,...]，fn 会把前缀维当 batch 一起处理
        """
        fn = partial(self.decode_and_resize_np,
                     resize_size=resize_size,
                     depth_resize_size=depth_resize_size)

        # 任务（通常是单帧，如果你的 task 也是序列，同样也能处理）
        frame["task"] = fn(frame["task"])

        # 观测：多个时间步、多张图，统统直接给 fn 就行
        frame["observation"] = fn(frame["observation"])

        return frame

    def sample_one_timestep_keep_obs_T1(self, traj: Dict[str, Any]) -> Dict[str, Any]:
        """
        随机选一个时间步 t：
        - observation 里的 ndarray: [T, ...] -> [1, ...]  (保留时间维，长度为 1)
        - 其它 key 里的 ndarray: [T, ...] -> [...]        (去掉时间维)
        """
        actions = np.asarray(traj["action"])
        T = actions.shape[0]
        t = np.random.randint(T)  # 随机选取时间索引

        def slice_tree(x, keep_time_axis: bool):
            if isinstance(x, dict):
                return {k: slice_tree(v, keep_time_axis) for k, v in x.items()}

            if isinstance(x, np.ndarray):
                # 只有第 0 维等于 T 的才视为“有时间维”的量
                if x.shape[0] == T:
                    if keep_time_axis:
                        # 保留时间维度: [T, ...] -> [1, ...]
                        return x[t:t + 1]
                    else:
                        # 去掉时间维度: [T, ...] -> [...]
                        return x[t]
                else:
                    return x

            return x  # 其它类型原样返回

        new_traj: Dict[str, Any] = {}
        for k, v in traj.items():
            if k == "observation":
                # observation 里的变量保留时间维 T=1
                new_traj[k] = slice_tree(v, keep_time_axis=True)
            else:
                # 其它变量去掉时间维
                new_traj[k] = slice_tree(v, keep_time_axis=False)

        return new_traj

    def sample_multi_timesteps_keep_obs_TK(
            self,
            traj: Dict[str, Any],
            k: int
    ) -> Dict[str, Any]:
        """
        一条轨迹里随机采 K 个时间步，保持时间维度长度 = K：
        - observation / action / task / dataset_name / absolute_action_mask 等
          只要第 0 维是 T，就用 idxs 做切片，变成 [K, ...]
        """
        actions = np.asarray(traj["action"])
        T = actions.shape[0]

        k = min(k, T)  # 不要超过trajectory长度
        idxs = np.random.choice(T, size=k, replace=False)  # [K]

        def slice_tree(x):
            if isinstance(x, dict):
                return {kk: slice_tree(vv) for kk, vv in x.items()}
            if isinstance(x, np.ndarray) and x.shape[0] == T:
                # 只对第 0 维是 T 的量做切片
                return x[idxs]
            return x

        new_traj: Dict[str, Any] = {}
        for key, value in traj.items():
            # 这些 key 里第 0 维通常是时间维
            if key in ("observation", "action", "task",
                       "dataset_name", "absolute_action_mask"):
                new_traj[key] = slice_tree(value)
            else:
                new_traj[key] = value

        return new_traj


    def __getitem__(self, idx):
        new_data_dict = dict()
        with h5py.File(self.file_paths[idx], 'r') as f:
            action = f["action"][...]
            proprio = f["observations"]["qpos"][...]
            timestep = np.arange(0, proprio.shape[0])
            image_primary = f["observations"]["images"]["cam_high"][...]
            image_left_wrist = f["observations"]["images"]["cam_left"][...]
            image_right_wrist = f["observations"]["images"]["cam_right"][...]
            pad_mask_dict = dict(
                image_primary=np.array([True for i in range(image_primary.shape[0])]),
                image_left_wrist=np.array([True for i in range(image_left_wrist.shape[0])]),
                image_right_wrist=np.array([True for i in range(image_right_wrist.shape[0])]),
                proprio=np.array([True for i in range(proprio.shape[0])]),
                timestep=np.array([True for i in range(timestep.shape[0])])
            )
            if action.shape[1] == 14:
                absolute_action_mask = np.array([[False] * 6 + [True] + [False] * 6 + [True] for i in range(action.shape[0])])
            if action.shape[1] == 7:
                absolute_action_mask = np.array([[False] * 6 + [True] for i in range(action.shape[0])])

            new_data_dict["absolute_action_mask"] = absolute_action_mask
            new_data_dict["observation"] = dict(
                image_primary = image_primary,
                image_left_wrist = image_left_wrist,
                image_right_wrist = image_right_wrist,
                proprio = proprio,
                timestep = timestep,
                pad_mask_dict=pad_mask_dict
            )
            new_data_dict["task"] =  dict(
                language_instruction = np.array([Path(self.file_paths[idx]).parent.name.replace("_", " ") for i in range(proprio.shape[0])]),
                pad_mask_dict=dict(
                    language_instruction=np.array([True for i in  range(proprio.shape[0])])
                )
            )
            new_data_dict["action"] = action
            new_data_dict["dataset_name"] = [self.dataset_name for i in range(action.shape[0])]

            new_data_dict = self.uniform_np(new_data_dict)
            new_data_dict = self.chunk_act_obs_np(
                traj=new_data_dict,
                window_size=1,
                future_action_window_size=NUM_ACTIONS_CHUNK-1)
            new_data_dict = self.apply_obs_transform_np(new_data_dict, self.resize_resolution, self.depth_resize_resolution)
            if self.train_mode and self.image_aug:
                new_data_dict = aug_np(new_data_dict, self.image_augment_kwargs)
            if self.num_samples_per_traj == 1:
                new_data_dict = self.sample_one_timestep_keep_obs_T1(new_data_dict)
            else:
                new_data_dict = self.sample_multi_timesteps_keep_obs_TK(
                    new_data_dict,
                    self.num_samples_per_traj
                )
            new_data_dict = self.batch_transform(new_data_dict)
            return new_data_dict

    def __len__(self) -> int:
        return len(self.file_paths)



def main():
    data_dir = "//datasets"
    data_name = "custom_dataset"
    dataset = HDF5Dataset(
        data_dir=data_dir,
        dataset_name=data_name,
        resize_resolution=(224, 224),
        depth_resize_resolution=(224, 224),
        image_augment_kwargs= dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )
    )
    for i in dataset:
        print(i)

if __name__ == "__main__":
    main()

