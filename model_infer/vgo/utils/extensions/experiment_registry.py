"""
实验配置哈希计算与 GitLab 注册

功能:
- 计算实验配置的稳定哈希 (arch_hash, hyper_hash, data_hash, exp_hash, run_hash)
- 注册实验配置到本地/远程 Git 仓库
- 根据哈希查询配置
- 支持 checkpoint 权重转换时的注册

使用方式:
    # 自动注册（在训练启动时）
    from vgo.utils.experiment_registry import compute_and_register
    exp_hash = compute_and_register(exp_config, engine_config)

    # 查询实验
    from vgo.utils.experiment_registry import get_registry
    config = get_registry().lookup("a3f9c8")

    # 注册权重转换
    from vgo.utils.experiment_registry import register_weight_conversion
    register_weight_conversion(source_hash, target_hash, conversion_info)
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    pass


# ==================== 数据结构 ====================


@dataclass
class ExperimentHash:
    """实验哈希结果"""

    hash_version: str = "v1"
    arch_hash: str = ""  # 模型结构 (DiT, VAE, LLM encoder) + 初始化权重路径
    hyper_hash: str = ""  # 训练超参 (optimizer, policy, precision) + parallelism
    data_hash: str = ""  # 数据配置
    exp_hash: str = ""  # 完整实验 (arch + hyper + data + lora) - 影响实验结果
    location_hash: str = ""  # 实验存储位置 (exp_root + name) - 不影响实验结果
    run_hash: str = ""  # 单次运行 (exp + location + seed + commit)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    def short_id(self, length: int = 8) -> str:
        """返回短 ID 用于显示"""
        return self.exp_hash[:length]

    def __str__(self) -> str:
        return (
            f"ExperimentHash(\n"
            f"  exp_hash={self.exp_hash},\n"
            f"  arch_hash={self.arch_hash},\n"
            f"  hyper_hash={self.hyper_hash},\n"
            f"  data_hash={self.data_hash},\n"
            f"  location_hash={self.location_hash},\n"
            f"  run_hash={self.run_hash}\n"
            f")"
        )


@dataclass
class RegistryConfig:
    """注册表配置"""

    enabled: bool = True  # 是否启用注册
    repo_url: str | None = None  # GitLab 仓库 URL（HTTPS 格式）
    local_path: str | None = None  # 本地路径，默认 ~/.experiment_registry
    branch: str = "main"  # Git 分支
    auto_push: bool = True  # 是否自动推送
    fail_on_error: bool = False  # 注册失败是否抛出异常（False 则仅警告）

    # Access Token（从环境变量读取，不要写在配置文件中！）
    # 设置环境变量 EXPERIMENT_REGISTRY_TOKEN 即可
    _access_token: str | None = field(default=None, repr=False)  # repr=False 避免打印时泄露

    def __post_init__(self):
        # 从环境变量读取默认值
        if self.repo_url is None:
            self.repo_url = os.environ.get(
                "EXPERIMENT_REGISTRY_URL", "https://gitlab.basemind.com/step_aigc/exp_zoo.git"
            )
        if self._access_token is None:
            self._access_token = os.environ.get("EXPERIMENT_REGISTRY_TOKEN")
        if self.local_path is None:
            self.local_path = os.environ.get("EXPERIMENT_REGISTRY_PATH", os.path.expanduser("~/.experiment_registry"))

    @property
    def has_token(self) -> bool:
        """是否配置了 access token"""
        return self._access_token is not None and len(self._access_token) > 0

    def get_gitlab_host(self) -> str | None:
        """从 repo_url 提取 GitLab 主机名"""
        if not self.repo_url:
            return None
        # https://gitlab.example.com/org/repo.git -> gitlab.example.com
        if self.repo_url.startswith("https://"):
            parts = self.repo_url[8:].split("/")
            return parts[0] if parts else None
        return None


# ==================== 序列化工具 ====================


def canonical_json(obj: Any) -> str:
    """
    稳定的 JSON 序列化（键排序、无空格、处理特殊类型）

    重要：这个函数的输出必须是确定性的，相同输入永远产生相同输出
    """

    def _normalize(o: Any) -> Any:
        if o is None:
            return None
        elif isinstance(o, dict):
            # 排序键，递归处理值，跳过 None 值
            return {k: _normalize(v) for k, v in sorted(o.items()) if v is not None}
        elif isinstance(o, (list, tuple)):
            return [_normalize(x) for x in o]
        elif isinstance(o, float):
            # 处理浮点数精度问题
            if o == int(o):
                return int(o)
            return round(o, 10)
        elif isinstance(o, (int, str, bool)):
            return o
        elif hasattr(o, "__dataclass_fields__"):
            return _normalize(asdict(o))
        elif hasattr(o, "to_dict"):
            return _normalize(o.to_dict())
        else:
            # 尝试转换为字符串
            return str(o)

    return json.dumps(_normalize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_hash(obj: Any, length: int = 16) -> str:
    """计算对象的 SHA256 哈希"""
    content = canonical_json(obj)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:length]


def _compute_file_hash(file_path: str, length: int = 16) -> str:
    """
    计算文件的 SHA256 哈希

    Args:
        file_path: 文件路径
        length: 返回的哈希长度（默认 16 字符）

    Returns:
        文件内容的 SHA256 哈希值（截取前 length 个字符）
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # 分块读取以支持大文件
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()[:length]


# ==================== 哈希计算器 ====================


class ExperimentHasher:
    """
    实验配置哈希计算器

    哈希层级:
    - arch_hash: 模型结构 (DiT params, VAE config, LLM encoder type)
    - hyper_hash: 训练超参 (optimizer, policy, precision, batch size, steps)
    - data_hash: 数据配置 (dataset, weights, augmentation)
    - exp_hash: 完整实验 = hash(arch + hyper + data + lora)
    - run_hash: 单次运行 = hash(exp + seed + git_commit)
    """

    @classmethod
    def compute(
        cls,
        exp_config: Any,
        engine_config: Any,
        data_config: Any | None = None,
        seed: int | None = None,
        commit_id: str | None = None,
        train_db_hash: str | None = None,
    ) -> ExperimentHash:
        """
        计算完整的实验哈希

        Args:
            exp_config: ExpConfig 实例或字典
            engine_config: EngineArgs 实例或字典
            data_config: DataConfigArgs 实例或字典（可选）
            seed: 随机种子（用于 run_hash）
            commit_id: Git commit ID（用于 run_hash）
            train_db_hash: train.db 文件的哈希值（可选，影响 data_hash 和 exp_hash）

        Returns:
            ExperimentHash 实例
        """
        # 1. Arch Hash (模型结构 + 初始化权重路径)
        arch_view = cls._extract_arch_view(engine_config)
        arch_hash = compute_hash(arch_view)

        # 2. Hyper Hash (训练超参 + parallelism)
        hyper_view = cls._extract_hyper_view(exp_config, engine_config)
        hyper_hash = compute_hash(hyper_view)

        # 3. Data Hash (数据配置 + train.db 哈希)
        data_view = cls._extract_data_view(engine_config, data_config, train_db_hash)
        data_hash = compute_hash(data_view)

        # 4. LoRA 配置
        lora_view = cls._extract_lora_view(engine_config)

        # 5. Exp Hash (完整实验 - 影响实验结果的所有配置)
        exp_view = {
            "arch": arch_view,
            "hyper": hyper_view,
            "data": data_view,
            "lora": lora_view,
        }
        exp_hash = compute_hash(exp_view)

        # 6. Location Hash (实验存储位置 - 不影响实验结果)
        location_view = cls._extract_location_view(exp_config)
        location_hash = compute_hash(location_view)

        # 7. Run Hash (单次运行 = exp + location + seed + commit)
        run_view = {
            "exp_hash": exp_hash,
            "location_hash": location_hash,
            "seed": seed,
            "commit_id": commit_id,
        }
        run_hash = compute_hash(run_view)

        return ExperimentHash(
            hash_version="v1",
            arch_hash=arch_hash,
            hyper_hash=hyper_hash,
            data_hash=data_hash,
            exp_hash=exp_hash,
            location_hash=location_hash,
            run_hash=run_hash,
        )

    @classmethod
    def _to_dict(cls, obj: Any) -> dict | None:
        """将对象转换为字典"""
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return dict(obj)

    @classmethod
    def _extract_arch_view(cls, engine_config: Any) -> dict:
        """提取模型结构配置（包含初始化权重路径）"""
        ec = cls._to_dict(engine_config) or {}
        pipe = ec.get("pipe", {})
        if hasattr(pipe, "__dataclass_fields__"):
            pipe = asdict(pipe)

        arch = {
            # 模型结构参数
            "dit": pipe.get("dit"),
            "llm_encoder_type": pipe.get("llm_encoder_type"),
            "llm_output_layer_index": pipe.get("llm_output_layer_index"),
            "llm_image_min_token": pipe.get("llm_image_min_token"),
            "llm_image_max_token": pipe.get("llm_image_max_token"),
            "max_length": pipe.get("max_length"),
            # 模型初始化权重路径（影响实验结果）
            "llm_model_path": pipe.get("llm_model_path"),
            "dit_path": pipe.get("dit_path"),
            "ae_path": pipe.get("ae_path"),
        }
        return arch

    @classmethod
    def _extract_hyper_view(cls, exp_config: Any, engine_config: Any) -> dict:
        """提取训练超参配置（包含 parallelism 和 policy）"""
        exp = cls._to_dict(exp_config) or {}
        ec = cls._to_dict(engine_config) or {}

        # 提取 parallelism 配置
        parallelism = exp.get("parallelism", {})
        if hasattr(parallelism, "__dataclass_fields__"):
            parallelism = asdict(parallelism)

        # 提取 policy 配置（确保嵌套 dataclass 被正确转换）
        policy = ec.get("policy", {})
        if hasattr(policy, "__dataclass_fields__"):
            policy = asdict(policy)

        # 提取 optim 配置
        optim = ec.get("optim", {})
        if hasattr(optim, "__dataclass_fields__"):
            optim = asdict(optim)

        # 提取 precision 配置
        precision = ec.get("precision", {})
        if hasattr(precision, "__dataclass_fields__"):
            precision = asdict(precision)

        hyper = {
            # 训练引擎
            "engine_target": exp.get("engine_target"),
            # 优化器配置
            "optim": optim,
            "policy": policy,
            "precision": precision,
            "model_precision": ec.get("model_precision"),
            # 训练配置
            "gradient_accumulation_steps": exp.get("gradient_accumulation_steps"),
            "micro_batch_size": exp.get("micro_batch_size"),
            "max_train_steps": exp.get("max_train_steps"),
            "ce_loss_weight": ec.get("ce_loss_weight"),
            # 并行配置（影响数值精度）
            "parallelism": parallelism,
        }
        return hyper

    @classmethod
    def _extract_data_view(cls, engine_config: Any, data_config: Any | None, train_db_hash: str | None = None) -> dict:
        """提取数据配置（包含 train.db 哈希）"""
        data_view = {}

        # 优先使用传入的 data_config
        if data_config is not None:
            data_view = cls._to_dict(data_config) or {}
        else:
            # 从 engine_config 中提取 data_config 路径
            ec = cls._to_dict(engine_config) or {}
            data_config_path = ec.get("data_config")

            if data_config_path and os.path.exists(data_config_path):
                # 读取数据配置文件内容并计算哈希
                try:
                    with open(data_config_path) as f:
                        import yaml

                        data_view = yaml.safe_load(f) or {}
                except Exception as e:
                    logger.warning(f"Failed to load data config from {data_config_path}: {e}")
                    data_view = {"path": data_config_path}
            else:
                data_view = {"path": data_config_path} if data_config_path else {}

        # 添加 train.db 哈希（影响 data_hash 和 exp_hash）
        if train_db_hash:
            data_view["train_db_hash"] = train_db_hash

        return data_view

    @classmethod
    def _extract_lora_view(cls, engine_config: Any) -> dict | None:
        """提取 LoRA 配置"""
        ec = cls._to_dict(engine_config) or {}
        pipe = ec.get("pipe", {})
        if hasattr(pipe, "__dataclass_fields__"):
            pipe = asdict(pipe)

        lora = pipe.get("lora")
        if lora is None:
            return None

        if hasattr(lora, "__dataclass_fields__"):
            return asdict(lora)
        return dict(lora) if lora else None

    @classmethod
    def _extract_location_view(cls, exp_config: Any) -> dict:
        """
        提取实验存储位置配置（不影响实验结果，但用于标识实验位置）

        这些字段用于区分同一实验配置的不同存储位置。
        """
        exp = cls._to_dict(exp_config) or {}

        location = {
            "exp_root": exp.get("exp_root"),
            "name": exp.get("name"),
        }
        return location


# ==================== GitLab 注册表 ====================


class GitLabExperimentRegistry:
    """
    基于 Git(Lab) 的实验注册表

    功能:
    - 注册实验配置到 Git 仓库
    - 根据 hash 查询配置
    - 支持本地/远程仓库
    - 支持权重转换记录
    - 支持 Access Token 认证
    """

    def __init__(self, config: RegistryConfig | None = None):
        """
        Args:
            config: 注册表配置，如果为 None 则使用默认配置
        """
        self.config = config or RegistryConfig()
        self.local_path = Path(self.config.local_path)  # type: ignore
        self._credential_file: Path | None = None

        if self.config.enabled:
            self._ensure_repo()
            self._setup_credentials()  # 在 repo 创建后设置凭证

    def _setup_credentials(self):
        """设置 Git 凭证（如果配置了 access token）"""
        if not self.config.has_token or not self.config.repo_url:
            return

        host = self.config.get_gitlab_host()
        if not host:
            return

        # 在 local_path 目录下创建 .git-credentials 文件（目录应该已存在）
        self._credential_file = self.local_path / ".git-credentials"

        # 写入凭证（格式：https://oauth2:TOKEN@host）
        credential_line = f"https://oauth2:{self.config._access_token}@{host}\n"
        self._credential_file.write_text(credential_line)
        self._credential_file.chmod(0o600)  # 只有 owner 可读写

        # 将 .git-credentials 添加到 .gitignore（避免意外提交）
        gitignore = self.local_path / ".gitignore"
        gitignore_content = gitignore.read_text() if gitignore.exists() else ""
        if ".git-credentials" not in gitignore_content:
            with open(gitignore, "a") as f:
                f.write("\n.git-credentials\n")

        logger.debug(f"Git credentials configured for {host}")

    def _run_git(self, args: list[str], **kwargs) -> subprocess.CompletedProcess:
        """
        运行 Git 命令，自动配置凭证

        Args:
            args: Git 命令参数（不包含 'git'）
            **kwargs: 传递给 subprocess.run 的其他参数

        Returns:
            subprocess.CompletedProcess
        """
        cmd = ["git"]

        # 如果配置了凭证文件，使用 -c 参数配置 credential helper
        if self._credential_file and self._credential_file.exists():
            cmd.extend(
                [
                    "-c",
                    f"credential.helper=store --file={self._credential_file}",
                ]
            )

        cmd.extend(args)

        # 默认参数
        kwargs.setdefault("cwd", self.local_path)
        kwargs.setdefault("capture_output", True)

        return subprocess.run(cmd, **kwargs)

    def _ensure_repo(self):
        """确保本地仓库存在"""
        if not self.local_path.exists():
            self.local_path.mkdir(parents=True)

            if self.config.repo_url:
                # Clone from remote（需要特殊处理，因为目录还不存在）
                try:
                    self._clone_repo()
                    logger.info("Cloned experiment registry from remote")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to clone registry: {e}, initializing local repo")
                    self._init_local_repo()
            else:
                self._init_local_repo()
        else:
            # 确保是 git 仓库
            if not (self.local_path / ".git").exists():
                self._init_local_repo()

        # 确保目录结构存在
        (self.local_path / "configs").mkdir(exist_ok=True)
        (self.local_path / "runs").mkdir(exist_ok=True)
        (self.local_path / "conversions").mkdir(exist_ok=True)

    def _clone_repo(self):
        """Clone 远程仓库"""
        cmd = ["git"]

        # 如果有 token，构建带凭证的 clone 命令
        if self.config.has_token and self.config.repo_url:
            host = self.config.get_gitlab_host()
            if host:
                # 使用临时凭证配置
                cmd.extend(
                    [
                        "-c",
                        f"credential.helper=!f() {{ echo username=oauth2; echo password={self.config._access_token}; }}; f",  # noqa: E501
                    ]
                )

        cmd.extend(["clone", self.config.repo_url, str(self.local_path)])

        subprocess.run(cmd, check=True, capture_output=True)

    def _init_local_repo(self):
        """初始化本地 Git 仓库"""
        self._run_git(["init"], check=True)

        # 创建初始提交
        readme = self.local_path / "README.md"
        readme.write_text("# Experiment Registry\n\nAuto-generated experiment configuration registry.\n")

        self._run_git(["add", "."], check=True)
        self._run_git(["commit", "-m", "Initial commit"], check=True)
        logger.info(f"Initialized local experiment registry at {self.local_path}")

    def _pull(self):
        """拉取最新（如果有远程）"""
        if not self.config.repo_url:
            return

        try:
            self._run_git(
                ["pull", "--rebase", "origin", self.config.branch],
                check=True,
                timeout=30,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.debug(f"Failed to pull from remote (will use local state): {e}")

    def _commit_and_push(self, message: str, tag: str | None = None):
        """提交并推送（如果配置了自动推送）"""
        try:
            self._run_git(["add", "."], check=True)

            # 检查是否有变更
            result = self._run_git(["diff", "--cached", "--quiet"])
            if result.returncode == 0:
                logger.debug("No changes to commit")
                return

            self._run_git(["commit", "-m", message], check=True)

            if tag:
                self._run_git(["tag", "-f", tag])

            if self.config.auto_push and self.config.repo_url:
                self._run_git(
                    ["push", "--tags", "origin", self.config.branch],
                    timeout=60,
                )
                logger.debug(f"Pushed to remote: {message}")

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Git operation failed: {e}")

    def register(
        self,
        exp_hash: ExperimentHash,
        config: dict,
        metadata: dict | None = None,
    ) -> str:
        """
        注册实验配置

        Args:
            exp_hash: 计算好的实验哈希
            config: 完整配置
            metadata: 额外元数据

        Returns:
            exp_hash.exp_hash
        """
        if not self.config.enabled:
            return exp_hash.exp_hash

        self._pull()

        hash_str = exp_hash.exp_hash
        config_path = self.local_path / "configs" / f"{hash_str}.json"

        # 如果已存在，跳过
        if config_path.exists():
            logger.info(f"Experiment {hash_str[:8]}... already registered")
            return hash_str

        # 保存配置快照
        snapshot = {
            "hash": exp_hash.to_dict(),
            "config": config,
            "metadata": {
                "registered_at": datetime.now().isoformat(),
                **(metadata or {}),
            },
        }
        config_path.write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # 更新索引
        self._update_index(exp_hash, metadata)

        # 提交
        self._commit_and_push(
            message=f"Register experiment {hash_str[:8]}",
            tag=f"exp/{hash_str}",
        )

        logger.info(f"Registered experiment: {hash_str[:8]}...")
        return hash_str

    def _update_index(self, exp_hash: ExperimentHash, metadata: dict | None):
        """更新索引文件"""
        index_path = self.local_path / "index.json"

        if index_path.exists():
            index = json.loads(index_path.read_text())
        else:
            index = {"version": "1", "experiments": {}, "conversions": []}

        index["experiments"][exp_hash.exp_hash] = {
            "arch_hash": exp_hash.arch_hash,
            "hyper_hash": exp_hash.hyper_hash,
            "data_hash": exp_hash.data_hash,
            "run_hash": exp_hash.run_hash,
            "config_path": f"configs/{exp_hash.exp_hash}.json",
            "registered_at": datetime.now().isoformat(),
            **(metadata or {}),
        }

        index_path.write_text(
            json.dumps(index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def lookup(self, exp_hash: str) -> dict | None:
        """
        根据哈希查询配置（支持前缀匹配）

        Args:
            exp_hash: 完整或部分哈希值

        Returns:
            配置快照字典，如果未找到返回 None
        """
        if not self.config.enabled:
            return None

        self._pull()

        config_path = self.local_path / "configs" / f"{exp_hash}.json"

        if not config_path.exists():
            # 尝试前缀匹配
            matches = list((self.local_path / "configs").glob(f"{exp_hash}*.json"))
            if len(matches) == 1:
                config_path = matches[0]
            elif len(matches) > 1:
                logger.warning(f"Multiple matches for {exp_hash}: {[m.stem for m in matches]}")
                return None
            else:
                return None

        return json.loads(config_path.read_text())

    def list_experiments(self, arch_hash: str | None = None) -> list[dict]:
        """
        列出实验

        Args:
            arch_hash: 如果指定，只返回相同架构的实验
        """
        index_path = self.local_path / "index.json"
        if not index_path.exists():
            return []

        index = json.loads(index_path.read_text())
        experiments = index.get("experiments", {})

        results = []
        for exp_hash, info in experiments.items():
            if arch_hash is None or info.get("arch_hash") == arch_hash:
                results.append({"exp_hash": exp_hash, **info})

        return results

    def register_run(
        self,
        run_hash: str,
        exp_hash: str,
        run_metadata: dict,
    ):
        """注册单次运行"""
        if not self.config.enabled:
            return

        run_path = self.local_path / "runs" / f"{run_hash}.json"
        run_data = {
            "run_hash": run_hash,
            "exp_hash": exp_hash,
            "registered_at": datetime.now().isoformat(),
            **run_metadata,
        }

        # 添加 KubeBrain 环境信息
        kubebrain_process_name = os.environ.get("KUBEBRAIN_PROCESS_NAME")
        kubebrain_rjob_name = os.environ.get("KUBEBRAIN_RJOB_NAME")
        brain_username = os.environ.get("BRAIN_USERNAME")

        if kubebrain_process_name:
            # rlaunch 环境
            if brain_username:
                run_data["user"] = brain_username
            run_data["env_name"] = kubebrain_process_name
            run_data["env_type"] = "rlaunch"
        elif kubebrain_rjob_name:
            # rjob 环境
            if brain_username:
                run_data["user"] = brain_username
            run_data["env_name"] = kubebrain_rjob_name
            run_data["env_type"] = "rjob"

        run_path.write_text(
            json.dumps(run_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._commit_and_push(f"Register run {run_hash[:8]}")

    def register_weight_conversion(
        self,
        source_hash: str | None,
        target_path: str,
        conversion_info: dict,
    ) -> str:
        """
        注册权重转换

        Args:
            source_hash: 源实验哈希（如果已知）
            target_path: 目标权重路径
            conversion_info: 转换信息（包含转换脚本、参数等）

        Returns:
            转换记录的哈希
        """
        if not self.config.enabled:
            return ""

        conversion_data = {
            "source_exp_hash": source_hash,
            "target_path": target_path,
            "converted_at": datetime.now().isoformat(),
            **conversion_info,
        }

        conversion_hash = compute_hash(conversion_data, length=12)
        conversion_path = self.local_path / "conversions" / f"{conversion_hash}.json"

        conversion_path.write_text(
            json.dumps(conversion_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # 更新索引
        index_path = self.local_path / "index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
        else:
            index = {"version": "1", "experiments": {}, "conversions": []}

        index.setdefault("conversions", []).append(
            {
                "conversion_hash": conversion_hash,
                "source_exp_hash": source_hash,
                "target_path": target_path,
                "converted_at": conversion_data["converted_at"],
            }
        )
        index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))

        self._commit_and_push(
            f"Register weight conversion {conversion_hash}",
            tag=f"conv/{conversion_hash}",
        )

        logger.info(f"Registered weight conversion: {conversion_hash}")
        return conversion_hash


# ==================== 全局实例和便捷函数 ====================

_default_registry: GitLabExperimentRegistry | None = None


def get_registry(config: RegistryConfig | None = None) -> GitLabExperimentRegistry:
    """获取默认注册表实例（单例模式）"""
    global _default_registry
    if _default_registry is None:
        _default_registry = GitLabExperimentRegistry(config)
    return _default_registry


def reset_registry():
    """重置默认注册表（用于测试）"""
    global _default_registry
    _default_registry = None


def compute_and_register(  # noqa: C901
    exp_config: Any,
    engine_config: Any,
    data_config: Any | None = None,
    registry_config: RegistryConfig | None = None,
    extra_metadata: dict | None = None,
) -> ExperimentHash | None:
    """
    计算哈希并注册（一站式接口）

    这是训练启动时调用的主入口。

    Args:
        exp_config: ExpConfig 实例
        engine_config: EngineArgs 实例
        data_config: 数据配置（可选）
        registry_config: 注册表配置（可选）
        extra_metadata: 额外元数据

    Returns:
        ExperimentHash 实例，如果注册被禁用或失败返回 None
    """
    # 检查是否启用
    if registry_config is not None and not registry_config.enabled:
        logger.debug("Experiment registry is disabled")
        return None

    try:
        # 加载 data_config 文件内容（如果 data_config 为 None）
        data_config_content = data_config
        data_config_path = None
        train_db_hash = None
        if data_config is None:
            # 从 engine_config 中提取 data_config 路径
            ec = engine_config
            if hasattr(ec, "__dataclass_fields__"):
                from dataclasses import asdict

                ec = asdict(ec)
            elif hasattr(ec, "to_dict"):
                ec = ec.to_dict()
            elif not isinstance(ec, dict):
                ec = dict(ec) if hasattr(ec, "__iter__") else {}

            data_config_path = ec.get("data_config")
            if data_config_path and os.path.exists(data_config_path):
                try:
                    import yaml

                    with open(data_config_path) as f:
                        data_config_content = yaml.safe_load(f)
                    logger.debug(f"Loaded data config from {data_config_path}")
                except Exception as e:
                    logger.warning(f"Failed to load data config from {data_config_path}: {e}")
                    data_config_content = {"_path": data_config_path, "_error": str(e)}

        # 计算 train.db 文件的哈希值
        if data_config_content and isinstance(data_config_content, dict):
            train_database = data_config_content.get("train_database")
            if train_database:
                train_db_path = os.path.join(train_database, "train.db")
                if os.path.exists(train_db_path):
                    try:
                        train_db_hash = _compute_file_hash(train_db_path)
                        logger.debug(f"Computed train.db hash: {train_db_hash}")
                    except Exception as e:
                        logger.warning(f"Failed to compute train.db hash: {e}")

        # 计算哈希
        commit_id = getattr(exp_config, "commit_id", None)
        exp_hash = ExperimentHasher.compute(
            exp_config=exp_config,
            engine_config=engine_config,
            data_config=data_config_content,
            seed=getattr(exp_config, "seed", None),
            commit_id=commit_id,
        )

        # 获取注册表
        registry = get_registry(registry_config)

        # 构建配置快照
        from dataclasses import asdict as dataclass_asdict

        # 敏感字段列表（不应保存到注册表）
        SENSITIVE_KEYS = {"_access_token", "access_token", "token", "password", "secret"}

        def safe_asdict(obj):
            if obj is None:
                return None
            if isinstance(obj, dict):
                return obj
            if hasattr(obj, "__dataclass_fields__"):
                return dataclass_asdict(obj)
            return dict(obj) if hasattr(obj, "__iter__") else str(obj)

        def remove_sensitive_keys(obj):
            """递归移除敏感字段"""
            if isinstance(obj, dict):
                return {k: remove_sensitive_keys(v) for k, v in obj.items() if k not in SENSITIVE_KEYS}
            elif isinstance(obj, list):
                return [remove_sensitive_keys(item) for item in obj]
            return obj

        config_snapshot = {
            "exp_config": remove_sensitive_keys(safe_asdict(exp_config)),
            "engine_config": remove_sensitive_keys(safe_asdict(engine_config)),
            "data_config": data_config_content,  # 使用加载的文件内容
            "data_config_path": data_config_path,  # 记录文件路径
            "train_db_hash": train_db_hash,  # train.db 文件的哈希值
            "commit_id": commit_id,  # 确保记录 commit_id
        }

        # 注册
        registry.register(
            exp_hash=exp_hash,
            config=config_snapshot,
            metadata={
                "name": getattr(exp_config, "name", "unknown"),
                "commit_id": commit_id,
                **(extra_metadata or {}),
            },
        )

        return exp_hash

    except Exception as e:
        error_msg = f"Failed to register experiment: {e}"
        if registry_config and registry_config.fail_on_error:
            raise RuntimeError(error_msg) from e
        logger.warning(error_msg)
        return None


def register_weight_conversion(
    source_hash: str | None,
    target_path: str,
    conversion_info: dict,
    registry_config: RegistryConfig | None = None,
) -> str | None:
    """
    注册权重转换（便捷函数）

    在进行模型权重转换时调用。

    Args:
        source_hash: 源实验的 exp_hash（如果已知）
        target_path: 转换后权重的保存路径
        conversion_info: 转换信息字典，建议包含:
            - script: 转换脚本路径
            - source_path: 源权重路径
            - conversion_type: 转换类型 (e.g., "lora_merge", "quantize", "prune")
            - parameters: 转换参数

    Returns:
        转换记录的哈希，如果失败返回 None
    """
    try:
        registry = get_registry(registry_config)
        return registry.register_weight_conversion(source_hash, target_path, conversion_info)
    except Exception as e:
        logger.warning(f"Failed to register weight conversion: {e}")
        return None


def lookup_experiment(exp_hash: str, registry_config: RegistryConfig | None = None) -> dict | None:
    """
    查询实验配置（便捷函数）

    Args:
        exp_hash: 实验哈希（支持前缀匹配）
        registry_config: 注册表配置

    Returns:
        配置快照字典
    """
    try:
        registry = get_registry(registry_config)
        return registry.lookup(exp_hash)
    except Exception as e:
        logger.warning(f"Failed to lookup experiment: {e}")
        return None
