import random
from typing import Any

import torch
from torchvision.transforms.functional import to_tensor
from vgo.data import DataTrackInfo
from vgo.data.processor.image import default_pil_resize, degrade_image, random_reference_pil_resize
from vgo.data.processor.image_shape_mapping import create_router_and_targetsizer
from vgo.data.processor.naive_collect import PackData
from vgo.data.vaultloader import TrainingSequence
from vgo.utils.transformers_compat import AutoProcessor


class NaiveSequenceCollateFn:
    def __init__(
        self,
        *,
        base_size: list[int],
        base_size_weights: list[float],
        enable_random_degrade=False,
    ):
        # FIXME: 重构分辨率采样的方式
        self.target_shape_mapping = create_router_and_targetsizer(
            [1 for _ in base_size],
            base_size=base_size,
            base_size_weights=base_size_weights,
            enable_multi_size=True,
            step_size=16,
        )
        self.enable_random_degrade = enable_random_degrade

    def convert_image_to_tensor(self, sequence: TrainingSequence):
        for image_idx, image in enumerate(sequence.image):
            sequence.image[image_idx] = to_tensor(image) * 2 - 1

        for image_idx, image in enumerate(sequence.ref_images):
            sequence.ref_images[image_idx] = to_tensor(image) * 2 - 1
        return sequence

    def resize_image_in_sequence(self, sequence: TrainingSequence, target_base_size):
        if target_base_size is None:
            target_base_size, _, _, _ = self.target_shape_mapping.target_shape(
                sequence.image[-1], return_base_size=True
            )
        for image_idx, image in enumerate(sequence.image):
            target_height, target_width = self.target_shape_mapping.get_suitable_image_shape(image, target_base_size)
            sequence.image[image_idx] = default_pil_resize(image, height=target_height, width=target_width)

        for image_idx, image in enumerate(sequence.ref_images):
            target_height, target_width = self.target_shape_mapping.get_suitable_image_shape(image, target_base_size)
            sequence.ref_images[image_idx] = random_reference_pil_resize(
                image, height=target_height, width=target_width, _random=self.random
            )

        return sequence

    @property
    def random(self):
        if not hasattr(self, "_random"):
            worker_info = torch.utils.data.get_worker_info()
            # FIXME: hard code seed
            seed = 42
            if worker_info is not None:
                worker_id = worker_info.id
                global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                seed = seed + (worker_info.num_workers * global_rank + worker_id)
            self._random = random.Random(seed)
        return self._random

    def random_degrade(self, sequence: TrainingSequence):
        sequence.ref_images = [degrade_image(x, _random=self.random) for x in sequence.ref_images]
        return sequence

    def process_images(self, sequence: TrainingSequence, sequence_id: Any):
        target_base_size = sequence_id[1] if isinstance(sequence_id, tuple) else None
        sequence = self.resize_image_in_sequence(sequence, target_base_size)

        # 设置随机降低质量
        if self.enable_random_degrade:
            sequence = self.random_degrade(sequence)

        sequence = self.convert_image_to_tensor(sequence)

        return sequence

    def __call__(self, _sequences: list[tuple[list[TrainingSequence], list[Any]]]):
        sequences = _sequences[0][0]
        sequence_ids = _sequences[0][1]

        sequences = [
            self.process_images(sequence, sequence_id) for sequence, sequence_id in zip(sequences, sequence_ids)
        ]

        pack_data = PackData(sequences=sequences)

        pack_data.data_track_info = [
            DataTrackInfo(
                _source=sequence.source,
                _url=sequence.uri,
                _sequence_id=sequence.sequence_id,
                _choice_id=sequence.choice_id,
                _target_height=sequence.image[-1].shape[1],
                _target_width=sequence.image[-1].shape[2],
            )
            for sequence in sequences
        ]
        return pack_data


if __name__ == "__main__":
    import os

    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoProcessor

    from vgo.data.processor.prefix_util import PrefixSetting
    from vgo.data.vaultloader import LoadBalancedVaultDataset
    from vgo.utils.dist_utils import ParallelDims, device_module, device_type, init_distributed

    model_path = "/mnt/sirui/model_zoo/Qwen2.5-VL-7B-Instruct/"
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=256 * 28 * 28,
        max_pixels=400 * 28 * 28,
    )

    PrefixSetting.set_PREFIX_MODE("qwen-image")
    pipeline = NaiveSequenceCollateFn(base_size=[1024], base_size_weights=[2.0], enable_random_degrade=True)

    def init_distributed_engine():
        device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(device)

        # init distributed
        world_size = int(os.environ["WORLD_SIZE"])

        parallel_dims = ParallelDims(
            dp=1,
            tp_w_sp=2,
            world_size=world_size,
        )

        init_distributed(
            init_timeout_seconds=300,
            dump_folder=None,
            trace_buf_size=1000,
            enable_cpu_offload=False,
        )

        # build meshes
        world_mesh = parallel_dims.build_mesh(device_type=device_type)
        return world_mesh

    world_mesh = init_distributed_engine()
    if world_mesh["tp_w_sp"].get_local_rank() != 0:
        import itertools

        train_dataloader = itertools.repeat(None)
    else:
        from omegaconf import OmegaConf

        config = OmegaConf.load("configs/data/example.yaml")
        dataset = LoadBalancedVaultDataset(
            "/mnt/sirui/test_vault/StepFlow-V2-10",
            source_weights=config.source_weights,
            base_size=[1024],
            base_size_weights=[2.0],
            max_length=4096 * 16,
            load_balance_buffer_factor=max(4, world_mesh["dp"].size()),
            dp_rank=world_mesh["dp"].get_local_rank(),
            dp_size=world_mesh["dp"].size(),
        )

        train_dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            sampler=None,
            num_workers=16,
            collate_fn=pipeline,
            pin_memory=True,
            in_order=True,
        )

    import time

    for i, _data in enumerate(tqdm(train_dataloader)):
        torch.distributed.breakpoint()
        time.sleep(1.0)
        i += 1
        if i == 100:
            break
