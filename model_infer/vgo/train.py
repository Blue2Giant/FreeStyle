import importlib
import json
import os
import pprint
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Union

import torch
import torch.distributed
from loguru import logger
from omegaconf import MISSING, OmegaConf
from torch.distributed.device_mesh import DeviceMesh
from torch.utils import tensorboard
from tqdm import tqdm

from vgo.train_engines import Engine, StepInfo
from vgo.train_engines.train_utils import TrainState
from vgo.utils.common_utils import DETERMINISTIC_MODE, GPUMemoryMonitor, get_commit_id_from_git_dir, listify, set_seed
from vgo.utils.dist_utils import ParallelDims, Parallelism, device_module, device_type, init_distributed
from vgo.utils.extensions.experiment_registry import RegistryConfig, compute_and_register, get_registry
from vgo.utils.timer import Timer

TIMER_STEP_GLOBAL = "step_global"
TIMER_STEP_EPILOGUE = "step_epilogue"
TIMER_CHECKPOINT = "checkpoint"


@dataclass
class TensorboardArgs:
    root_dir: str | None = None
    log_dir: str | None = None
    run_name: str | None = None

    def update_log_dir(self, exp: "ExpConfig"):
        self.run_name = exp.name
        self.log_dir = os.path.join(self.root_dir, self.run_name) if self.root_dir is not None else exp.logging_dir


@dataclass
class LoggingArgs:
    log_freq: int = 10  # Log every freq * global_steps (i.e. after gradient accumulation)
    image_freq: int = 1000  # Log validation images every N global steps
    report_to: str = "tensorboard"
    tensorboard: TensorboardArgs = field(default_factory=TensorboardArgs)


@dataclass
class SaveEvery:
    every: int = 2500  # Save every N *global* steps
    keep: int = 0  # 0 means keep all checkpoints


@dataclass
class CheckpointArgs:
    dump: SaveEvery = field(default_factory=SaveEvery)
    path: str = ""


@dataclass
class DataRecordArgs:
    dump: SaveEvery = field(default_factory=lambda: SaveEvery(100, 0))
    path: str = ""
    enable: bool = True


@dataclass
class ExpConfig:
    exp_root: str = MISSING
    name: str = MISSING
    seed: int | None = 42 * 42
    allow_tf32: bool = True
    gradient_accumulation_steps: int = 1
    non_activation_checkpointing_every: int = -1  # enable checkpointing for all layers
    micro_batch_size: int | float = 8
    model_precision: str = "bf16"  # "fp32", "fp16", "bf16"

    max_train_steps: int = 50000
    resume_from_checkpoint: str | None = "latest"  # Path or "latest" or None

    environ: dict | None = None  # Environment variables to set
    policy: dict | None = None

    logging: LoggingArgs = field(default_factory=LoggingArgs)
    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    parallelism: Parallelism = field(default_factory=Parallelism)

    engine_target: str = "projects.step1x_image.engines.diffusion_accelerate"
    engine_config: dict = field(default_factory=dict)

    # 实验注册配置
    registry: RegistryConfig = field(default_factory=RegistryConfig)

    # Internal state, set by __post_init__ or runtime
    exp_dir: str = field(init=False)
    logging_dir: str = field(init=False)
    date: str = field(init=False)
    id: str = field(init=False)
    global_batch_size: int | float = field(init=False)
    data_record: DataRecordArgs = field(default_factory=DataRecordArgs)
    commit_id: str | None = field(init=False)

    def __post_init__(self):
        # Create paths
        self.exp_dir = os.path.join(self.exp_root, self.name)
        self.logging_dir = os.path.join(self.exp_dir, "logs")
        self.checkpoint.path = (
            os.path.join(self.exp_dir, "checkpoints") if len(self.checkpoint.path) == 0 else self.checkpoint.path
        )

        self.data_record.path = (
            os.path.join(self.exp_dir, "data_record") if len(self.data_record.path) == 0 else self.data_record.path
        )
        if self.data_record.dump.every > self.checkpoint.dump.every:
            logger.warning(
                f"{self.data_record.dump=} is smaller than {self.checkpoint.dump.every=}, use self.checkpoint.dump.every instead."  # noqa: E501
            )
            self.data_record.dump.every = min(self.data_record.dump.every, self.checkpoint.dump.every)

        # Create unique identifiers
        self.date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.id = f"{self.name}_{self.date}"
        # Derived config
        # Note: Assumes accelerator is available later to get world size
        self.global_batch_size = MISSING

        # 自动获取 commit id
        self.commit_id = get_commit_id_from_git_dir()
        if self.commit_id is None:
            logger.warning("Failed to get commit id.")

    def update_global_batch_size(self, dp_size: int):
        self.global_batch_size = self.micro_batch_size * self.gradient_accumulation_steps * dp_size
        logger.info(f"Global batch size: {self.global_batch_size}")

    def validate(self):
        assert Path(self.checkpoint.path).name == "checkpoints"  # Safety check
        assert self.checkpoint.dump.every > 0
        assert self.gradient_accumulation_steps > 0
        assert self.micro_batch_size > 0

    def dump(self, path: str | Path, log_config=True):
        """Saves the config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        yaml_dump = OmegaConf.to_yaml(OmegaConf.structured(self))
        with open(path, "w") as f:
            if log_config:
                logger.info("--- Experiment Configuration ---")
                # Indent the yaml dump for better readability in logs
                indented_yaml = "\n".join("  " + line for line in yaml_dump.splitlines())
                logger.info(f"\n{indented_yaml}")
                logger.info("-----------------------------")
            f.write(yaml_dump)

    def setup_environment(self):
        """Sets environment variables defined in EnvironmentArgs."""
        if self.environ is None:
            return
        for key, value in self.environ.items():
            os.environ[key] = str(value)
            logger.debug(f"Set environment variable: {key}={value}")


class TensorBoardTracker:
    def __init__(self, run_name: str, logging_dir: Union[str, os.PathLike], **kwargs):
        self.run_name = run_name
        self.logging_dir = logging_dir
        self.on_main_process = torch.distributed.get_rank() == 0

        if self.on_main_process:
            self.writer = tensorboard.SummaryWriter(self.logging_dir, **kwargs)
            logger.debug(f"Initialized TensorBoard project {self.run_name} logging to {self.logging_dir}")

    @property
    def tracker(self):
        return self.writer

    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment. Stores the
        hyperparameters in a yaml file for future use.

        Args:
            values (Dictionary `str` to `bool`, `str`, `float` or `int`):
                Values to be stored as initial hyperparameters as key-value pairs. The values need to have type `bool`,
                `str`, `float`, `int`, or `None`.
        """
        if self.on_main_process:
            self.writer.add_hparams(values, metric_dict={})
            self.writer.flush()

    def log(self, values: dict, step: int | None = None, flush=True, **kwargs):
        """
        Logs `values` to the current run.

        Args:
            values (Dictionary `str` to `str`, `float`, `int` or `dict` of `str` to `float`/`int`):
                Values to be logged as key-value pairs. The values need to have type `str`, `float`, `int` or `dict` of
                `str` to `float`/`int`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to either `SummaryWriter.add_scaler`,
                `SummaryWriter.add_text`, or `SummaryWriter.add_scalers` method based on the contents of `values`.
        """
        if self.on_main_process:
            values = listify(values)
            for k, v in values.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(k, v, global_step=step, **kwargs)
                elif isinstance(v, str):
                    self.writer.add_text(k, v, global_step=step, **kwargs)
                elif isinstance(v, dict):
                    self.writer.add_scalars(k, v, global_step=step, **kwargs)
            if flush:
                self.writer.flush()

    def log_images(self, values: dict, step: int | None, flush=True, **kwargs):
        """
        Logs `images` to the current run.

        Args:
            values (Dictionary `str` to `List` of `np.ndarray` or `PIL.Image`):
                Values to be logged as key-value pairs. The values need to have type `List` of `np.ndarray` or
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to the `SummaryWriter.add_image` method.
        """
        if self.on_main_process:
            for k, v in values.items():
                self.writer.add_images(k, v, global_step=step, **kwargs)
            if flush:
                self.writer.flush()

    def finish(self):
        """
        Closes `TensorBoard` writer
        """
        if self.on_main_process:
            self.writer.close()
            logger.debug("TensorBoard writer closed")


def resume_engine(
    engine: Engine,
    train_state: TrainState,
    checkpoint_dir: os.PathLike | str | None = None,
    path: str | None = "latest",
):
    """Handles resuming training from a checkpoint."""
    load_path = None
    resume_opt = path
    use_resume = False
    checkpoint_loaded = False

    if resume_opt:
        if resume_opt == "latest":
            assert checkpoint_dir is not None, f"`checkpoint_dir` can not be None if {resume_opt=}"
            if os.path.isdir(checkpoint_dir):
                # Find the latest checkpoint directory (e.g., "checkpoint-5000")
                dirs = []
                for d in os.listdir(checkpoint_dir):
                    checkpoint_path = os.path.join(checkpoint_dir, d)
                    if not os.path.isdir(checkpoint_path):
                        continue
                    if not d.startswith("checkpoint-"):
                        continue
                    suffix = d.split("-", 1)[1]
                    if not suffix.isdigit():
                        continue
                    dirs.append(d)
                if dirs:
                    dirs.sort(key=lambda x: int(x.split("-")[1]))
                    load_path = os.path.join(checkpoint_dir, dirs[-1])
                    use_resume = True
                    logger.info(f"Found latest checkpoint: {load_path}")
                else:
                    logger.info(f"No checkpoints found in {checkpoint_dir} to resume from.")
            else:
                logger.warning(f"Checkpoint directory {checkpoint_dir} does not exist. Cannot resume 'latest'.")
        else:
            # Specific path provided
            if Path(resume_opt).exists() and Path(resume_opt).is_dir():
                load_path = resume_opt
                use_resume = True
                logger.info(f"Attempting to resume from specified checkpoint: {load_path}")
            else:
                logger.warning(
                    f"Specified checkpoint path {resume_opt} does not exist or is not a directory. Starting fresh."
                )

        if load_path:
            try:
                engine.load_checkpoint(load_path)
                checkpoint_loaded = True
            except Exception as e:
                logger.error(f"Failed to load checkpoint from {load_path}: {e}")
                logger.exception(e)
                logger.warning("Starting training from scratch due to loading error.")
            if use_resume and checkpoint_loaded:
                train_state_path = os.path.join(load_path, "train_state.pth")
                if os.path.exists(train_state_path):
                    train_state.load_state_dict(torch.load(train_state_path))
                    engine.set_init_train_state(train_state)
                    logger.success(f"Continue Training from global step: {train_state.global_step}")
                else:
                    logger.warning(f"Missing train state file at {train_state_path}. Continue from step 0.")
            elif use_resume and not checkpoint_loaded:
                logger.warning("Skip train_state resume because engine checkpoint did not load successfully.")
        else:
            logger.info("No valid checkpoint path found. Starting training from scratch.")
    else:
        logger.info("resume_from_checkpoint path is None. Starting training from scratch.")


def build_engine(exp: ExpConfig, device_mesh: DeviceMesh) -> Engine:
    engine_impl = importlib.import_module(exp.engine_target)

    engine_config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(engine_impl.EngineArgs), OmegaConf.create(exp.engine_config))
    )

    assert exp.seed is not None
    engine = engine_impl.Engine.build(
        config=engine_config,
        device_mesh=device_mesh,
        dataloader_seed=exp.seed,
        micro_batch_size=exp.micro_batch_size,
        use_data_recorder=exp.data_record.enable,
        gradient_accumulation_steps=exp.gradient_accumulation_steps,
        non_activation_checkpointing_every=exp.non_activation_checkpointing_every,
    )
    return engine


def training_loop(exp: ExpConfig, engine: Engine, tracker: TensorBoardTracker | None, exp_hash=None):  # noqa: C901
    timers = {n: Timer(name=n) for n in [TIMER_STEP_GLOBAL, TIMER_STEP_EPILOGUE, TIMER_CHECKPOINT]}
    timers.update(engine.get_timer())

    train_state = TrainState()

    resume_engine(engine, train_state, exp.checkpoint.path, exp.resume_from_checkpoint)

    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {exp.micro_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {exp.global_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {exp.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {exp.max_train_steps}")
    logger.info(f"Device memory usage: {GPUMemoryMonitor(device_type)}")

    logger.info(f"{train_state=}")

    pbar = None
    if torch.distributed.get_rank() == 0:
        pbar = tqdm(
            initial=train_state.global_step,
            total=exp.max_train_steps,
            desc=exp.id,
            unit="step",
        )

    batch_iterator = iter(engine.batch_generator())
    last_checkpoint_step = train_state.global_step

    while train_state.global_step < exp.max_train_steps:
        timers[TIMER_STEP_GLOBAL].start()
        engine.gc_handler.run(train_state.global_step)
        info: StepInfo = engine.train_one_step(batch_iterator)
        train_state.global_step += 1
        timers[TIMER_STEP_GLOBAL].stop()

        # compute Tokens/sec
        token_per_sec = {}
        for k, v in info.scalar_metrics.items():
            if k.endswith("token_count"):
                v = v * exp.gradient_accumulation_steps * engine.device_mesh["dp"].size() / engine.device_mesh.size()
                v = v / timers[TIMER_STEP_GLOBAL].get_last()  # type: ignore
                token_per_sec["token_count/" + k + "_per_sec"] = v
        # 去掉 TPS
        info.scalar_metrics = {k: v for k, v in info.scalar_metrics.items() if not k.endswith("token_count")}

        with timers[TIMER_STEP_EPILOGUE]:
            if pbar:
                pbar.update(1)

            if (
                train_state.global_step % exp.checkpoint.dump.every == 0
                and train_state.global_step > last_checkpoint_step
            ) or train_state.global_step == 1:
                with timers[TIMER_CHECKPOINT]:
                    checkpoint_folder = (
                        Path(exp.checkpoint.path) / f"checkpoint-{train_state.global_step}"
                    ).as_posix()
                    engine.save_checkpoint(checkpoint_folder)
                    # only save train state on rank 0
                    if torch.distributed.get_rank() == 0:
                        os.makedirs(
                            checkpoint_folder,
                            exist_ok=True,
                        )
                        torch.save(
                            train_state.state_dict(),
                            (
                                Path(exp.checkpoint.path) / f"checkpoint-{train_state.global_step}" / "train_state.pth"
                            ).as_posix(),
                        )
                        # 保存实验哈希
                        if exp_hash is not None:
                            hash_file = Path(checkpoint_folder) / "exp_hash.json"
                            hash_file.write_text(json.dumps(exp_hash.to_dict(), indent=2))

                last_checkpoint_step = train_state.global_step

            if exp.data_record.enable and train_state.global_step % exp.data_record.dump.every == 0:
                engine.save_data_record(Path(exp.data_record.path).as_posix())

            if train_state.global_step % exp.logging.log_freq == 0:
                if pbar:
                    step_global_last = timers[TIMER_STEP_GLOBAL].get_last(n=exp.logging.log_freq)
                    step_epilogue_last = timers[TIMER_STEP_EPILOGUE].get_last(n=exp.logging.log_freq)
                    lines = [
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} - ",
                        f"global_step={train_state.global_step}",
                        f"lr={info.lr:.8f}",
                        f"loss={info.loss:.4f}",
                        f"grad_norm={info.global_grad_norm:.4f}",
                        (
                            f"TIMER_STEP_GLOBAL {step_global_last:.4f}s"
                            if step_global_last is not None
                            else "TIMER_STEP_GLOBAL N/A"
                        ),
                        (
                            f"TIMER_STEP_EPILOGUE {step_epilogue_last:.4f}s"
                            if step_epilogue_last is not None
                            else "TIMER_STEP_EPILOGUE N/A"
                        ),
                    ]
                    pbar.write(" | ".join(lines))

            if tracker is not None:
                times = {
                    f"time/{n}": timers[n].acess_last()
                    for n in timers
                    if n != TIMER_STEP_EPILOGUE and timers[n].num_times
                }
                tracker.log(times, step=train_state.global_step, flush=False)

                if len(token_per_sec) > 0:
                    tracker.log(token_per_sec, step=train_state.global_step, flush=False)

                if info.images is not None:
                    tracker.log_images(info.images, step=train_state.global_step, flush=False)

                # 此时timers[TIMER_STEP_EPILOGUE]还没stop，拿到的是上一个step的耗时
                if timers[TIMER_STEP_EPILOGUE].num_times:
                    tracker.log(
                        {f"time/{TIMER_STEP_EPILOGUE}": timers[TIMER_STEP_EPILOGUE].acess_last()},
                        step=train_state.global_step - 1,
                        flush=False,
                    )

                values = dict(
                    lr=info.lr, global_grad_norm=info.global_grad_norm, loss=info.loss, **info.scalar_metrics
                )
                tracker.log(values, step=train_state.global_step, flush=False)

    logger.info("Training completed.")
    if train_state.global_step > last_checkpoint_step:
        engine.save_checkpoint((Path(exp.checkpoint.path) / f"checkpoint-{train_state.global_step}").as_posix())

    # 关闭tracker，确保所有数据写入完成
    if tracker is not None:
        logger.info("Closing TensorBoard tracker and waiting for all writes to complete...")
        tracker.finish()


def init_logger(log_folder: Path, global_rank, exp_id: str, log_in_console=True):
    logger.remove()
    if log_in_console and global_rank == 0:
        logger.add(sys.stderr, level="DEBUG")
    logger.add(log_folder / f"{exp_id}_{global_rank}.log", level="DEBUG", backtrace=True, diagnose=True)


def init_distributed_engine(exp: ExpConfig):
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
    # Device has to be set before creating TorchFT manager.
    device_module.set_device(device)

    # init distributed
    world_size = int(os.environ["WORLD_SIZE"])

    parallel_dims = ParallelDims(
        dp=exp.parallelism.data_parallel_replicate_degree,
        tp_w_sp=exp.parallelism.tensor_parallel_with_sequenc_parallel_degree,
        world_size=world_size,
    )

    init_distributed(
        init_timeout_seconds=300,
        dump_folder=Path(Path(exp.exp_dir) / "dump"),
        trace_buf_size=1000,
        enable_cpu_offload=False,
    )

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    return world_mesh


@logger.catch()
def train(exp: ExpConfig):  # noqa: C901
    exp.setup_environment()

    device_mesh = init_distributed_engine(exp)

    if DETERMINISTIC_MODE:
        logger.warning("Use Deterministic Mode For Debugging")
        torch.use_deterministic_algorithms(True)

    if exp.seed is not None:
        logger.info(f"Setting random seed to {exp.seed}")
        set_seed(exp.seed)
    else:
        logger.warning("No random seed set.")

        exp.seed = random.randint(2**8, 2**24)
        logger.info(f"Random seed set to {exp.seed}")
        set_seed(exp.seed)

    if exp.allow_tf32:
        if device_type == "cuda":
            logger.info("Allowing TF32 matrix multiplication.")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            logger.info(f"TF32 is CUDA-only, ignored on {device_type=}.")
    else:
        if device_type == "cuda":
            logger.info("Disabling TF32 matrix multiplication.")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    init_logger(Path(exp.logging_dir), device_mesh.get_rank(), exp.id, log_in_console=True)

    gpu_memory_monitor = GPUMemoryMonitor(device_type)
    logger.info(f"Device memory usage: {gpu_memory_monitor}")

    logger.info(f"{torch.distributed.get_rank()=} TP: {device_mesh['tp_w_sp']}")

    exp.update_global_batch_size(device_mesh["dp"].size())

    if torch.distributed.get_rank() == 0:
        os.makedirs(exp.exp_dir, exist_ok=True)
        os.makedirs(exp.logging_dir, exist_ok=True)
        os.makedirs(exp.checkpoint.path, exist_ok=True)
        exp.dump(Path(exp.exp_dir) / f"config--{exp.id}.yaml")

    engine = build_engine(exp, device_mesh)

    if hasattr(engine, "set_logdir"):
        engine.set_logdir(exp.logging_dir, exp.id)

    # 自动注册实验（仅在 rank 0 执行）
    exp_hash = None
    if torch.distributed.get_rank() == 0 and exp.registry.enabled:
        try:
            exp_hash = compute_and_register(
                exp_config=exp,
                engine_config=engine.config if hasattr(engine, "config") else exp.engine_config,
                registry_config=exp.registry,
                extra_metadata={
                    "exp_dir": exp.exp_dir,
                    "date": exp.date,
                },
            )
            if exp_hash is not None:
                logger.success(f"Experiment registered with hash: {exp_hash.short_id()}")
                logger.info(f"Full experiment hash:\n{exp_hash}")
                # 保存 exp_hash 到 exp_dir
                hash_file = Path(exp.exp_dir) / "exp_hash.json"
                hash_file.write_text(json.dumps(exp_hash.to_dict(), indent=2))
                # 注册本次运行
                registry = get_registry(exp.registry)
                registry.register_run(
                    run_hash=exp_hash.run_hash,
                    exp_hash=exp_hash.exp_hash,
                    run_metadata={
                        "name": exp.name,
                        "exp_dir": exp.exp_dir,
                        "seed": exp.seed,
                        "commit_id": exp.commit_id,
                        "date": exp.date,
                        "location_hash": exp_hash.location_hash,
                    },
                )
        except Exception as e:
            if exp.registry.fail_on_error:
                raise
            logger.warning(f"Experiment registration failed (non-blocking): {e}")

    if exp.logging.tensorboard is not None:
        exp.logging.tensorboard.update_log_dir(exp)
        assert exp.logging.tensorboard.log_dir is not None
        assert exp.logging.tensorboard.run_name is not None
        tracker = TensorBoardTracker(
            run_name=exp.logging.tensorboard.run_name,
            logging_dir=exp.logging.tensorboard.log_dir,
            max_queue=1000,
            flush_secs=120,
        )
        logger.info(f"TensorBoardTracker initialized: {tracker}")
    else:
        tracker = None

    training_loop(exp=exp, engine=engine, tracker=tracker, exp_hash=exp_hash)


@logger.catch()
def main(*omega_options, config_file: str | None = None, dump_config_as_yaml: str | None = None):
    config = OmegaConf.structured(ExpConfig)

    if config_file is not None:
        file_config = OmegaConf.load(config_file)
        logger.success(f"Config loaded from {Path(config_file).absolute()}")
        config = OmegaConf.merge(config, file_config)

    omega_options = [str(o) for o in omega_options]
    if len(omega_options) > 0:
        cli_config = OmegaConf.from_cli(omega_options)
        logger.info(f"will merge {pprint.pformat(OmegaConf.to_container(cli_config))} to config")
        config = OmegaConf.merge(config, cli_config)

    if dump_config_as_yaml is not None:
        if len(config.engine_config) == 0:
            engine_impl = importlib.import_module(config.engine_target)
            config.engine_config = asdict(engine_impl.EngineArgs())

        OmegaConf.save(config, dump_config_as_yaml)
        logger.success(f"Config dumped to {dump_config_as_yaml}")
        return

    exp: ExpConfig = OmegaConf.to_object(config)
    exp.validate()

    logger.info(f"`{exp.id}` start training")
    train(exp)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
