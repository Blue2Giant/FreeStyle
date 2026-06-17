import datetime
import io
import random
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

import duckdb
import megfile
import PIL.Image
import torch
from loguru import logger
from pydantic import BaseModel
from torch.utils.data.dataset import IterableDataset

from vgo.data.processor.image_shape_mapping import create_router_and_targetsizer
from vgo.data.sampler import AttrToLenIndexSampler, IndexSampler, SampleInstance


@dataclass
class DataConfigArgs:
    train_database: str
    source_weights: dict[str, float]
    base_size: list[int]
    base_size_weights: list[float]
    enable_random_degrade: bool = False
    text_dropout_rate: float = 0.1


class Image(BaseModel):
    id: UUID
    width: int
    height: int
    index: str


class Text(BaseModel):
    id: UUID
    content: str
    language: Literal["cn", "en", "zh"] | None
    index: str


class ModalInstance(BaseModel):
    type: Literal["image", "text"]
    index: str
    require_loss: bool
    ref_index: str | None = None


class SequenceChoice(BaseModel):
    choice: list[ModalInstance]


class TrainingSequence(BaseModel):
    message_formatter: str
    image: list[Any]
    ref_images: list[Any]
    text: list[str]
    require_loss: list[bool]
    task_type: Literal["t2i", "edit", "customize"]
    droppable: list[bool] | None = None
    sequence_id: UUID
    source: str
    uri: str
    sys_prompt: str | None
    choice_id: int

    @staticmethod
    def recompute_message_formatter(split_message):
        text_idx = 0
        image_idx = 0
        message_formatter = ""
        for x in split_message:
            if x[0] == "text":
                message_formatter += "<" + x[0] + "_" + str(text_idx) + ">"
                text_idx += 1
            else:
                message_formatter += "<" + x[0] + "_" + str(image_idx) + ">"
                image_idx += 1
        return message_formatter

    @staticmethod
    def split_message(format_message: str):
        pattern = r"(<[^<>|]+>)"
        parts = re.split(pattern, format_message)
        parts = [x for x in parts if len(x) > 0]
        parts = [(split_x[0], int(split_x[1])) for x in parts for split_x in [x[1:-1].split("_")]]
        return parts

    def dropout_text(
        self,
        _random: random.Random,
        rate=0.1,
    ):
        # 注意，这里由于是 causal 的，如果前面的文本不见了，后面被预测的文本最好可以不被预测
        # FIXME: 增加 droppable 的支持
        split_message = self.split_message(self.message_formatter)
        text_idx_map = {}
        text_idx = 0
        for i, modal_instance_i in enumerate(split_message):
            if "text" in modal_instance_i:
                text_idx_map[text_idx] = i
                text_idx += 1

        for text_idx in range(len(self.text)):
            if self.droppable is not None:
                if not self.droppable[text_idx]:
                    continue
            if _random.random() < rate:
                for _ in range(text_idx, len(self.text)):
                    max_key = list(text_idx_map.keys())[-1]
                    if self.droppable is not None:
                        if not self.droppable[max_key]:
                            text_idx_map.pop(max_key)
                            continue
                    split_message.pop(text_idx_map[max_key])
                    self.require_loss.pop(text_idx_map[max_key])
                    self.text.pop(max_key)
                    text_idx_map.pop(max_key)
                break

        self.message_formatter = self.recompute_message_formatter(split_message)


class PackSequence(BaseModel):
    sequence_id: UUID
    vault_path: str
    uri: str
    source: str
    task_type: Literal["t2i", "edit", "customize"]
    create_time: datetime.datetime
    images: list[Image]
    texts: list[Text]
    sequence_choices: list[SequenceChoice]
    choices_weights: list[float] | None = None

    def convert_to_train_sequence(self, sequence_choice_idx) -> TrainingSequence:
        import os

        # vault (step-vault) is only needed for training-time data loading, not inference.
        from vault.schema import ID

        keep_text_idx = os.environ.get("VGO_KEEP_TEXT_INDEX", "").strip() or None

        if keep_text_idx:
            for new_idx, ch in enumerate(self.sequence_choices):
                text_indices = [m.index for m in ch.choice if m.type == "text"]
                if keep_text_idx in text_indices:
                    sequence_choice_idx = new_idx
                    break

        choice = self.sequence_choices[sequence_choice_idx]

        message_formatter = ""
        image = []
        text = []
        ref_images = []
        require_loss = []
        modal_type_cnt = defaultdict(int)

        sys_prompt = None

        text_lookup = {x.index: x.content for x in self.texts}

        for _modal_idx, modal_instance in enumerate(choice.choice):
            # FIXME: 强制要求 sys_prompt 的 index 为 sys_prompt
            if modal_instance.index != "sys_prompt":
                if (
                    keep_text_idx
                    and modal_instance.type == "text"
                    and modal_instance.index != keep_text_idx
                ):
                    continue
                message_formatter += f"<{modal_instance.type}_{modal_type_cnt[modal_instance.type]}>"
                modal_type_cnt[modal_instance.type] += 1
            else:
                sys_prompt = text_lookup.get(modal_instance.index, "")
                continue

            if modal_instance.type == "image":
                modal_id = next(ID.from_uuid(x.id) for x in self.images if x.index == modal_instance.index)
                image.append([self.vault_path, modal_id])
                if modal_instance.ref_index is not None:
                    ref_modal_id = next(ID.from_uuid(x.id) for x in self.images if x.index == modal_instance.ref_index)
                    ref_images.append([self.vault_path, ref_modal_id])
                else:
                    ref_images.append([self.vault_path, modal_id])
            elif modal_instance.type == "text":
                text.append(text_lookup.get(modal_instance.index, ""))

            require_loss.append(modal_instance.require_loss)

        return TrainingSequence(
            message_formatter=message_formatter,
            image=image,
            ref_images=ref_images,
            text=text,
            require_loss=require_loss,
            task_type=self.task_type,
            source=self.source,
            uri=self.uri,
            sequence_id=self.sequence_id,
            sys_prompt=sys_prompt,
            choice_id=sequence_choice_idx,
        )


def create_vault_index_sampler(
    vault_train_path,
):
    with duckdb.connect(vault_train_path, read_only=True) as conn:
        source_list = conn.execute(
            """
            SELECT DISTINCT source
            FROM sequences_data;
            """
        ).fetchall()
        source_list = [x[0] for x in source_list]
        logger.info(f"Found source list {source_list}")

        source_sequence_id_list = {}
        for source_name in source_list:
            id_list = conn.execute(
                f"""
                SELECT
                    s.sequence_id,
                    CASE
                        WHEN s.choices_weights IS NOT NULL
                        THEN s.choices_weights
                        ELSE ( -- This subquery will now return exactly one row with a single list value
                            SELECT LIST(1.0) -- Aggregate all the '1.0' values into a single list: [1.0, 1.0, ...]
                            FROM range( LENGTH(s.sequence_choices) )
                        )
                    END AS choices_weights
                FROM sequences_data AS s
                WHERE source = '{source_name}';
                """,
            ).fetchall()

            source_sequence_id_list[source_name] = [
                SampleInstance(sample_id=x[0], choices_weights=x[1]) for x in id_list
            ]

    return IndexSampler(source_sequence_id_list)


def get_source_list(vault_train_path):
    with duckdb.connect(vault_train_path, read_only=True) as conn:
        source_list = conn.execute(
            """
            SELECT DISTINCT source
            FROM sequences_data;
            """
        ).fetchall()
        source_list = [x[0] for x in source_list]
        source_list.sort()

        return source_list


def create_load_balanced_vault_index_sampler(
    vault_train_path,
    max_length,
    source_weights: dict[str, float] | None,
    base_size: list[int],
    base_size_weights: list[float],
    load_balance_buffer_factor: int = 1,
    seed: int = 42,
):
    target_shape_mapping = create_router_and_targetsizer(
        [1 for _ in base_size],
        base_size=base_size,
        base_size_weights=base_size_weights,
        enable_multi_size=True,
        step_size=16,
    )

    def map_to_length(image_info: Any, _random: random.Random):
        task_type, height, width, choice_image_count = image_info
        _, allowed_base_sizes, _, _ = target_shape_mapping.target_shape((height, width), return_base_size=True)

        if isinstance(allowed_base_sizes, list):
            _base_size_weights = [
                weight for size, weight in zip(base_size, base_size_weights) if size in allowed_base_sizes
            ]
            chosen_base_size = _random.choices(allowed_base_sizes, _base_size_weights)[0]
        else:
            chosen_base_size = allowed_base_sizes

        t_h, t_w = target_shape_mapping.get_suitable_image_shape((height, width), chosen_base_size)

        length = t_h * t_w // 16 // 16
        if choice_image_count is None:
            choice_image_count = 2 if task_type == "edit" else 1
        length *= max(int(choice_image_count), 1)

        return length, chosen_base_size

    with duckdb.connect(vault_train_path, read_only=True) as conn:
        source_list = conn.execute(
            """
            SELECT DISTINCT source
            FROM sequences_data;
            """
        ).fetchall()
        source_list = [x[0] for x in source_list]
        logger.info(f"Found source list {source_list}")

        df = conn.execute(
            """
            SELECT
                s.source,
                s.sequence_id,
                CASE
                    WHEN s.choices_weights IS NOT NULL
                    THEN s.choices_weights
                    ELSE REPEAT(
                        [CAST(1.0 AS DOUBLE)],
                        LENGTH(s.sequence_choices)
                    )
                END AS choices_weights,
                s.task_type,
                (ARRAY_FILTER(s.images, x -> x.index = 'target'))[1].height AS target_height,
                (ARRAY_FILTER(s.images, x -> x.index = 'target'))[1].width AS target_width,
                LIST_MAX(
                    LIST_TRANSFORM(
                        s.sequence_choices,
                        c -> LENGTH(LIST_FILTER(c.choice, m -> m.type = 'image'))
                    )
                ) AS choice_image_count
            FROM
                sequences_data AS s,
            """
        ).df()

        # FIXME: speed up, the line below takes about 28s. Please use df sampler instead

        def make_group_for_load_balanced(g):
            arr = g[
                ["sequence_id", "choices_weights", "task_type", "target_height", "target_width", "choice_image_count"]
            ].to_numpy()

            return [
                (
                    SampleInstance(sample_id=sequence_id, choices_weights=choices_weights),
                    task_type,
                    target_height,
                    target_width,
                    choice_image_count,
                )
                for sequence_id, choices_weights, task_type, target_height, target_width, choice_image_count in arr
            ]

        source_sequence_id_attr_list = {source: make_group_for_load_balanced(g) for source, g in df.groupby("source")}

        if source_weights is None:
            source_weights = dict.fromkeys(source_sequence_id_attr_list, 1.0)

        total_weights = sum(source_weights.values())  # type: ignore
        source_weights = {k: v / total_weights for k, v in source_weights.items()}  # type: ignore

        assert all(x in source_list for x in source_weights), (
            f"Missing data source weights for {[x for x in source_weights if x not in source_list]}"
        )
        assert all(x in source_weights for x in source_list), (
            f"Unexpected data source weights {[x for x in source_list if x not in source_weights]}"
        )

        for source_name in source_sequence_id_attr_list:
            logger.info(
                f"Got Source `{source_name:>30}` with {len(source_sequence_id_attr_list[source_name]):>10,} Sequences, sampling ratio: {source_weights[source_name] * 100:,.2f}%."  # noqa: E501 # type: ignore
            )
        logger.info(
            f"Got {len(source_sequence_id_attr_list)} Sources with {sum([len(x) for x in source_sequence_id_attr_list.values()]):,}"  # noqa: E501
        )

        logger.info(f"Sampling Resolution: {dict(zip(base_size, base_size_weights))}")

    return AttrToLenIndexSampler(
        source_sequence_id_attr_list,  # type: ignore
        map_to_length,
        max_length=max_length,
        load_balance_buffer_factor=load_balance_buffer_factor,
        sample_weights_dict=source_weights,
        seed=seed,
    )


def from_image_bytes_to_pil_image(image_data: bytes):
    return PIL.Image.open(io.BytesIO(image_data)).convert("RGB")


class VaultSequenceLoader:
    def __init__(self, vault_train_folder: str, text_dropout_rate: float = 0.1):
        self.vault_train_folder = vault_train_folder
        self.text_dropout_rate = text_dropout_rate
        self._random: random.Random | None = None

    @property
    def random(self) -> random.Random:
        if self._random is None:
            worker_info = torch.utils.data.get_worker_info()
            # FIXME: hard code seed
            seed = 42
            if worker_info is not None:
                worker_id = worker_info.id
                global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                seed = seed + (worker_info.num_workers * global_rank + worker_id)
            self._random = random.Random(seed)
        return self._random

    def get_pack_sequence_from_ids(self, sequence_ids: list[tuple[UUID, int]]) -> list[TrainingSequence]:
        # vault (step-vault) is only needed for training-time data loading, not inference.
        from vault.backend.lance import LanceTaker
        from vault.schema import ID

        with duckdb.connect(megfile.smart_path_join(self.vault_train_folder, "train.db"), read_only=True) as conn:
            samples = conn.execute(
                """
                SELECT s
                FROM sequences_data AS s
                WHERE s.sequence_id IN ?;
                """,
                [tuple(x[0] for x in sequence_ids)],
            ).fetchall()

            train_sequence_list: list[TrainingSequence] = [None for _ in range(len(sequence_ids))]

            _sequence_list = [PackSequence(**sample[0]) for sample in samples]
            sequence_ids_to_sample_index = {x.sequence_id: i for i, x in enumerate(_sequence_list)}
            for i, seq_id in enumerate(sequence_ids):
                sequence = _sequence_list[sequence_ids_to_sample_index[seq_id[0]]]
                train_sequence_list[i] = sequence.convert_to_train_sequence(seq_id[1])

        images_to_be_fetch = defaultdict(set[ID])
        for sequence in train_sequence_list:
            for image in sequence.image:
                images_to_be_fetch[image[0]].add(image[1])
            for image in sequence.ref_images:
                images_to_be_fetch[image[0]].add(image[1])

        pil_images = {}
        for k, v in images_to_be_fetch.items():
            images = LanceTaker.by_ids(megfile.smart_path_join(k, "images"), list(v), columns=["id", "image"])

            pil_images.update({(k, x["id"]): from_image_bytes_to_pil_image(x["image"]) for x in images.to_pylist()})

        for sequence in train_sequence_list:
            sequence.image = [pil_images[(x[0], x[1].to_uuid())] for x in sequence.image]
            sequence.ref_images = [pil_images[(x[0], x[1].to_uuid())] for x in sequence.ref_images]
            # FIXME: 随机性应该也需要加到 sampler 里面才合适，能够做到吗？
            sequence.dropout_text(self.random, rate=self.text_dropout_rate)

        return train_sequence_list


class LoadBalancedVaultDataset(IterableDataset):
    def __init__(
        self,
        vault_train_folder: str,
        source_weights: dict[str, float] | None,
        base_size: list[int],
        base_size_weights: list[float],
        max_length=1,
        load_balance_buffer_factor: int = 1,
        dp_rank=0,
        dp_size: int = 1,
        seed: int = 42,
        post_processor: Callable | None = None,
        only_return_indices: bool = False,
        logfile: str | None = None,
        text_dropout_rate: float = 0.1,
    ):
        self.vault_train_folder = vault_train_folder
        self.vault_sequence_loader = VaultSequenceLoader(
            self.vault_train_folder,
            text_dropout_rate=text_dropout_rate,
        )
        self.index_sampler = create_load_balanced_vault_index_sampler(
            megfile.smart_path_join(vault_train_folder, "train.db"),
            source_weights=source_weights,
            max_length=max_length,
            base_size=base_size,
            base_size_weights=base_size_weights,
            load_balance_buffer_factor=load_balance_buffer_factor,
            seed=seed,
        )
        self.source_list = get_source_list(megfile.smart_path_join(vault_train_folder, "train.db"))
        self.batch_index = -1
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.only_return_indices = only_return_indices

        if load_balance_buffer_factor != self.dp_size:
            logger.warning(
                f"⚠️ 你正在使用负载均衡的 Dataloader，{load_balance_buffer_factor=} 应当和 {self.dp_size=} 相等，否则无法取得等价训练效果。"  # noqa: E501
                "注意，如果 load_balance_buffer_factor 较小，可能无法实现负载均衡。"
                "如果你正在使用 `tools/data_analysis/summarize_dataloader.py` 做数据分布分析，请忽略这一条警告信息。"
            )

        assert 0 <= self.dp_rank < self.dp_size

        self.post_processor = post_processor

        if logfile is not None:
            self.error_logger = logger.bind(
                dataset=__name__,
            )
            self.error_logger.add(
                logfile,
                level="ERROR",
                enqueue=True,
                filter=lambda record: "dataset" in record["extra"],
                format="{time} | {level: <8} | {process} | {name}:{function} - {message}",
            )
        else:
            self.error_logger = logger

        self.index_sampler_iter = iter(self.index_sampler)

    def set_logfile(self, logfile: str):
        self.error_logger = logger.bind(
            dataset=__name__,
        )
        self.error_logger.add(
            logfile,
            level="ERROR",
            enqueue=True,
            filter=lambda record: "dataset" in record["extra"],
            format="{time} | {level: <8} | {process} | {name}:{function} - {message}",
        )

    def iter_batch_index(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            worker_rank = 0
            worker_number = 1
        else:  # in a worker process
            worker_rank = worker_info.id * self.dp_size + self.dp_rank
            worker_number = worker_info.num_workers * self.dp_size
        logger.info(f"Launching dataloader worker {worker_rank=} from {worker_number} workers")
        while True:
            data_indices = next(self.index_sampler_iter)
            self.batch_index += 1
            if self.batch_index % worker_number == worker_rank:
                yield data_indices

    def resume_from(self, start_index: int, dp_size: int):
        logger.info(f"正在尝试从 {start_index} 步重启 Dataloader.")
        for _ in range(start_index * dp_size):
            _ = next(self.index_sampler_iter)
            self.batch_index += 1

    def __iter__(self):
        batch_index_iterator = iter(self.iter_batch_index())
        while True:
            data_indices = next(batch_index_iterator)

            if self.only_return_indices:
                yield data_indices
                continue

            try:
                training_sequences = self.vault_sequence_loader.get_pack_sequence_from_ids(
                    [x[0] for x in data_indices]
                )
            except Exception as _e:
                self.error_logger.exception(f"Failed to load data from indices: {data_indices}")
                continue

            if self.post_processor is not None:
                self.post_processor(data_indices, training_sequences)

            yield training_sequences, data_indices


class VaultDataset(IterableDataset):
    def __init__(
        self,
        vault_train_folder: str,
        batch_size=1,
        dp_rank=0,
        dp_size: int = 1,
        post_processor: Callable | None = None,
        only_return_indices: bool = False,
        logfile: str | None = None,
    ):
        self.vault_train_folder = vault_train_folder
        self.vault_sequence_loader = VaultSequenceLoader(self.vault_train_folder)
        self.index_sampler = create_vault_index_sampler(megfile.smart_path_join(vault_train_folder, "train.db"))
        self.batch_size = batch_size
        self.batch_index = -1
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.only_return_indices = only_return_indices
        self.post_processor = post_processor

        if logfile is not None:
            self.error_logger = logger.bind(
                dataset=__name__,
            )
            self.error_logger.add(
                logfile,
                level="ERROR",
                enqueue=True,
                filter=lambda record: "dataset" in record["extra"],
                format="{time} | {level: <8} | {process} | {name}:{function} - {message}",
            )
        else:
            self.error_logger = logger

    def set_logfile(self, logfile: str):
        self.error_logger = logger.bind(
            dataset=__name__,
        )
        self.error_logger.add(
            logfile,
            level="ERROR",
            enqueue=True,
            filter=lambda record: "dataset" in record["extra"],
            format="{time} | {level: <8} | {process} | {name}:{function} - {message}",
        )

    def resume_from(self, start_index: int, dp_size: int):
        logger.info(f"正在尝试从 {start_index} 步重启 Dataloader.")
        for _ in range(start_index * dp_size):
            _ = self.index_sampler.get_indices(batch_size=self.batch_size)
            self.batch_index += 1

    def iter_batch_index(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            worker_rank = 0
            worker_number = 1
        else:  # in a worker process
            worker_rank = worker_info.id * self.dp_size + self.dp_rank
            worker_number = worker_info.num_workers * self.dp_size
        logger.info(f"Launching dataloader worker {worker_rank=} from {worker_number} workers")
        while True:
            data_indices = self.index_sampler.get_indices(batch_size=self.batch_size)
            self.batch_index += 1
            if self.batch_index % worker_number == worker_rank:
                yield data_indices

    def __iter__(self):
        batch_index_iterator = iter(self.iter_batch_index())
        while True:
            data_indices = next(batch_index_iterator)

            if self.only_return_indices:
                yield data_indices
                continue

            try:
                training_sequences = self.vault_sequence_loader.get_pack_sequence_from_ids(data_indices)
            except Exception as _e:
                self.error_logger.exception(f"Failed to load data from indices: {data_indices}")
                continue

            if self.post_processor is not None:
                self.post_processor(data_indices, training_sequences)

            # FIXME: 这里逻辑有些问题，按道理应当根据 indices 中的一些 meta 信息处理好数据后再吐出去
            # 暂时懒得改了
            yield training_sequences, data_indices


if __name__ == "__main__":
    index_sampler = create_vault_index_sampler("/mnt/sirui/test_vault/train_test/train.db")
    index_list = index_sampler.get_indices(batch_size=32)
    loader = VaultSequenceLoader("/mnt/sirui/test_vault/train_test/")
    loader.get_pack_sequence_from_ids(index_list)
    dataset = VaultDataset("/mnt/sirui/test_vault/train_test/", batch_size=8)

    for _data in dataset:
        import pdb

        pdb.set_trace()
