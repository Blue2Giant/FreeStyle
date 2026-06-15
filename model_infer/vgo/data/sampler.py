import heapq
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass(slots=True)
class SampleInstance:
    sample_id: Any
    choices_weights: list[float]

    def get_key(self, _random: random.Random):
        idx = _random.choices(list(range(len(self.choices_weights))), weights=self.choices_weights)[0]
        return self.sample_id, idx


@dataclass
class Bucket:
    key: str
    _epoch_idx: int
    _sample_idx: int
    _id_list: list[SampleInstance]
    _init: bool = False

    def __post_init__(self):
        assert not self._init, "self._init should be False after initialization"

    def __len__(self):
        return len(self._id_list)

    def regret(self):
        self._sample_idx -= 1
        assert self._sample_idx < len(self._id_list) and self._sample_idx >= 0

    def get_next(self, _random: random.Random):
        if not self._init:
            _random.shuffle(self._id_list)
            self._init = True

        if self._sample_idx < len(self._id_list):
            sampled_id = self._id_list[self._sample_idx]
            self._sample_idx += 1
        else:
            self._epoch_idx += 1
            _random.shuffle(self._id_list)
            sampled_id = self._id_list[0]
            self._sample_idx = 1

        return sampled_id.get_key(_random)


class IndexSampler:
    sample_weights: dict[str, float]
    sample_buckets: dict[str, Bucket]
    _random: random.Random | None
    _step: int
    _seed: int

    def __init__(
        self,
        id_list_dict: dict[str, list[SampleInstance]],
        sample_weights_dict: dict[str, float] | None = None,
        step_index=0,
        seed=42,
    ):
        self.sample_buckets = {}
        if sample_weights_dict is not None:
            self.sample_weights = sample_weights_dict
        else:
            self.sample_weights = dict.fromkeys(id_list_dict, 1.0)

        self._step = step_index
        for k, v in id_list_dict.items():
            self.sample_buckets[k] = Bucket(
                key=k,
                _epoch_idx=0,
                _sample_idx=0,
                _id_list=v,
            )

        self._seed = seed
        self._random = None

    @property
    def random(self) -> random.Random:
        if self._random is None:
            self._random = random.Random(self._seed)
        return self._random

    def get_indices(self, *, batch_size):
        bucket_keys = tuple(self.sample_buckets.keys())
        bucket_weights = tuple(self.sample_weights.values())

        selected_bucket_keys = []
        batch_len = 0
        while len(selected_bucket_keys) < batch_size:
            next_buckets = self.random.choices(bucket_keys, bucket_weights)[0]
            selected_bucket_keys.append(self.sample_buckets[next_buckets].get_next(self.random))
            batch_len += 1

        return selected_bucket_keys


class AttrToLenIndexSampler:
    sample_weights: dict[str, float]
    sample_buckets: dict[str, Bucket]
    max_length: int
    id_to_attr_map: dict[Any, Any]
    attr_to_len_func: Callable[[Any, random.Random], tuple[int, Any]]
    _random: random.Random | None
    _step: int
    _seed: int

    def __init__(
        self,
        id_attr_list_dict: dict[str, list[tuple[SampleInstance, ...]]],
        attr_to_len_func: Callable[[Any, random.Random], tuple[int, Any]],
        max_length: int,
        load_balance_buffer_factor: int = 1,
        sample_weights_dict: dict[str, float] | None = None,
        seed: int = 42,
        step_index=0,
    ):
        self.sample_buckets = {}
        if sample_weights_dict is not None:
            self.sample_weights = sample_weights_dict
        else:
            self.sample_weights = dict.fromkeys(id_attr_list_dict, 1.0)

        self._step = step_index

        self.id_to_attr_map = {}
        for k, v in id_attr_list_dict.items():
            self.sample_buckets[k] = Bucket(
                key=k,
                _epoch_idx=0,
                _sample_idx=0,
                _id_list=[x[0] for x in v],
            )
            self.id_to_attr_map.update({x[0].sample_id: x[1:] for x in v})

        self.attr_to_len_func = attr_to_len_func
        self.load_balance_buffer_factor = load_balance_buffer_factor
        self.max_length = max_length
        self._random = None
        self._seed = seed

    @property
    def random(self) -> random.Random:
        if self._random is None:
            self._random = random.Random(self._seed)
        return self._random

    def load_balance_indices(self, instance_lens: list[int]):
        sorted_instances = sorted(enumerate(instance_lens), key=lambda x: x[1], reverse=True)

        buckets = [[] for _ in range(self.load_balance_buffer_factor)]
        bucket_lengths = [(0, i) for i in range(self.load_balance_buffer_factor)]

        heapq.heapify(bucket_lengths)
        for index, length in sorted_instances:
            min_bucket_length, min_bucket_index = heapq.heappop(bucket_lengths)
            buckets[min_bucket_index].append(index)
            heapq.heappush(bucket_lengths, (length + min_bucket_length, min_bucket_index))

        assert all(len(x) > 0 for x in buckets), "存在一个桶为空，会导致训练出现问题"

        self.random.shuffle(buckets)

        return buckets

    def __iter__(self):
        bucket_keys = tuple(self.sample_weights.keys())
        bucket_weights = tuple(self.sample_weights.values())

        while True:
            selected_bucket_keys = []
            attr_keys = []
            instance_len = []
            current_len = 0

            while True:
                next_buckets = self.random.choices(bucket_keys, bucket_weights)[0]
                selected_bucket_keys.append(self.sample_buckets[next_buckets].get_next(self.random))
                current_len_i, attr_key = self.attr_to_len_func(
                    self.id_to_attr_map[selected_bucket_keys[-1][0]], self.random
                )
                instance_len.append(current_len_i)
                attr_keys.append(attr_key)

                current_len += current_len_i

                if current_len > self.max_length * self.load_balance_buffer_factor:
                    # 超过最大允许长度，弹出一个
                    last_key = selected_bucket_keys.pop()
                    instance_len.pop()
                    attr_keys.pop()

                    # 队列中仍然存在，并且数量超过负载均衡的桶的数量
                    if len(attr_keys) >= 1 and self.load_balance_buffer_factor <= len(attr_keys):
                        self.sample_buckets[next_buckets].regret()
                        break
                    elif len(attr_keys) == 0:
                        logger.warning(f"单个样本过大，超过当前允许的最大长度。{next_buckets=}，{last_key=}")
                    else:
                        logger.warning(
                            f"Batch 中负载超过设置值，但是 batch 中的样本数量为 {len(attr_keys)}"
                            f"，不够 {self.load_balance_buffer_factor=} 个桶来分配。"
                        )

            if self.load_balance_buffer_factor > 1:
                for indices in self.load_balance_indices(instance_len):
                    yield list(zip([selected_bucket_keys[x] for x in indices], [attr_keys[x] for x in indices]))
            else:
                yield list(zip(selected_bucket_keys, attr_keys))
