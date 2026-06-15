from typing import Literal

from loguru import logger

ENCODE_PREFIX = """Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:
- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.
- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.\n
Here are examples of how to transform or refine prompts:
- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.
- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.\n
Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:
User Prompt:"""  # noqa: E501

EDIT_PREFIX = """Given a reference image and a user editing instruction.
You need to understand the information of the reference image and the content of the editing instruction,
and then generate a detailed description of the edited target image based on the reference image and the editing instruction.\u0020
The description should include the requirements in the editing instruction and try to closely match the content of the target image.
The editing instruction is:"""  # noqa: E501

QWENIMAGE_PREFIX = """Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:"""  # noqa: E501

QWENIMAGE_EDIT_PREFIX = """Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."""  # noqa: E501


class PrefixSetting:
    _instance = None
    _is_init = False

    _PREFIX_MODE = "step1x"
    _LQ_COUNT: int = 16

    _PREFIX_STRING_LEN: list[int] | None = None
    _PREFIX_TOKEN_LEN: list[int] | None = None
    _PREFIX_PROMPTS: dict[str, str] | None = None
    _PREFIX_TOKEN_LEN_DICT: dict[Literal["t2i", "edit", "customize"], int] | None = None
    _PREFIX_STRING_LEN_DICT: dict[Literal["t2i", "edit", "customize"], int] | None = None
    _PREFIX_TASK_TYPE_IDX: dict[str, int] | None = None
    _PREFIX_TOKEN_LEN_SUM: int | None = None

    @staticmethod
    def is_init():
        return PrefixSetting._is_init

    @property
    def PREFIX_MODE(self) -> str:
        if not PrefixSetting._is_init:
            logger.warning(
                f"LLMLQSetting is not mannually initialized, use default LQ count = {self._LQ_COUNT} instead."
            )
            PrefixSetting._is_init = True
        return self._PREFIX_MODE

    @property
    def PREDEFINED_PREFIX(self) -> list[str]:
        return ["t2i", "edit"]

    @property
    def PREFIX_STRING_LEN_DICT(self) -> dict[Literal["t2i", "edit", "customize"], int]:
        assert self.is_init()

        if self._PREFIX_STRING_LEN_DICT is None:
            if self._PREFIX_MODE == "step1x":
                self._PREFIX_STRING_LEN_DICT = {
                    "t2i": 1133,
                    "edit": 524,
                    "customize": 0,
                }
            else:  # qwen-image
                self._PREFIX_STRING_LEN_DICT = {
                    "t2i": 180,
                    "edit": 351,
                    "customize": 0,
                }

        return self._PREFIX_STRING_LEN_DICT

    @property
    def PREFIX_STRING_LEN(self) -> list[int]:
        assert self.is_init()

        if self._PREFIX_STRING_LEN is None:
            if self._PREFIX_MODE == "step1x":
                self._PREFIX_STRING_LEN = [
                    1133,  # "t2i"
                    524,  # "edit"
                    0,
                ]
            else:  # qwen-image
                self._PREFIX_STRING_LEN = [
                    180,  # "t2i"
                    351,  # "edit"
                    0,
                ]

        return self._PREFIX_STRING_LEN

    @property
    def PREFIX_TOKEN_LEN(self) -> list[int]:
        assert self.is_init()

        if self._PREFIX_TOKEN_LEN is None:
            if self._PREFIX_MODE == "step1x":
                self._PREFIX_TOKEN_LEN = [
                    217,  # "t2i"
                    90,  # "edit"
                    0,
                ]
            else:  # qwen-image
                self._PREFIX_TOKEN_LEN = [
                    34,  # "t2i"
                    64,  # "edit"
                    0,
                ]

        return self._PREFIX_TOKEN_LEN

    @property
    def PREFIX_PROMPTS(self) -> dict[str, str]:
        assert self.is_init()

        if self._PREFIX_PROMPTS is None:
            if self._PREFIX_MODE == "step1x":
                self._PREFIX_PROMPTS = {
                    "t2i": ENCODE_PREFIX,
                    "edit": EDIT_PREFIX,
                    "customize": "",
                }
            else:  # qwen-image
                self._PREFIX_PROMPTS = {
                    "t2i": QWENIMAGE_PREFIX,
                    "edit": QWENIMAGE_EDIT_PREFIX,
                    "customize": "",
                }

        return self._PREFIX_PROMPTS

    @property
    def PREFIX_TOKEN_LEN_DICT(self) -> dict[Literal["t2i", "edit", "customize"], int]:
        assert self.is_init()

        if self._PREFIX_TOKEN_LEN_DICT is None:
            if self._PREFIX_MODE == "step1x":
                self._PREFIX_TOKEN_LEN_DICT = {
                    "t2i": 217,
                    "edit": 90,
                    "customize": 0,
                }

            else:  # qwen-image
                self._PREFIX_TOKEN_LEN_DICT = {
                    "t2i": 34,
                    "edit": 64,
                    "customize": 0,
                }

        return self._PREFIX_TOKEN_LEN_DICT

    @property
    def PREFIX_TOKEN_LEN_SUM(self) -> int:
        assert self.is_init()
        if self._PREFIX_TOKEN_LEN_SUM is None:
            self._PREFIX_TOKEN_LEN_SUM = sum(self.PREFIX_TOKEN_LEN)
        return self._PREFIX_TOKEN_LEN_SUM

    @property
    def PREFIX_TASK_TYPE_IDX(self) -> dict[str, int]:
        assert self.is_init()
        if self._PREFIX_TASK_TYPE_IDX is None:
            self._PREFIX_TASK_TYPE_IDX = {k: i for i, k in enumerate(self.PREFIX_TOKEN_LEN_DICT)}

        return self._PREFIX_TASK_TYPE_IDX

    @PREFIX_MODE.setter
    def PREFIX_MODE(self, value):
        raise AttributeError("PREFIX_MODE is read-only")

    @staticmethod
    def set_PREFIX_MODE(PREFIX_MODE: Literal["step1x", "qwen-image"]):
        if PrefixSetting._is_init:
            raise AttributeError("PrefixSetting set_PREFIX_MODE cannot be called twice")
        PrefixSetting._PREFIX_MODE = PREFIX_MODE
        PrefixSetting._is_init = True

    def get_prefix_message(self, task_type: Literal["t2i", "edit", "customize"], sys_prompt=""):
        if sys_prompt:
            messages = [{"role": "system", "content": sys_prompt}]
            assert task_type not in self.PREDEFINED_PREFIX
        else:
            if self.PREFIX_MODE == "step1x":
                messages = []
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{self.PREFIX_PROMPTS[task_type]} "},  # type: ignore
                        ],
                    }
                )
            else:
                messages = [{"role": "system", "content": self.PREFIX_PROMPTS[task_type]}]
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": ""},  # type: ignore
                        ],
                    }
                )
        return messages

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
