import functools
import operator
import re

import numpy as np


def split_string_in_quotation(s: str, prefix_len: int) -> tuple[list[str], list[bool]]:  # noqa: C901
    """
    将字符串s根据中英文双引号进行分割。

    边界情况:

    1. 如果一个引号(中/英文) 被开启但直到字符串末尾也未被闭合,
      那么该起始引号及其后的所有内容都将被
      视为普通的、在引号之外的文本。

    Args:
        s (str): 输入的字符串。
        prefix_len (int): 前缀长度。在该长度之前的引号不作为分割符。

    Returns:
        tuple[list[str], list[bool]]: 分割后的字符串列表和对应的布尔标记列表。
    """
    result = []
    result_in_quotation = []

    state = "OUTSIDE"
    buffer = ""  # 用于累积引号外的内容

    quote_content_buffer = []

    for idx, char in enumerate(s):
        if idx < prefix_len:
            buffer += char
            continue

        if state == "OUTSIDE":
            if char == '"':
                result.append(buffer + '"')
                result_in_quotation.append(False)
                buffer = ""
                state = "IN_ENGLISH_QUOTES"
            elif char == "“":
                if buffer:
                    result.append(buffer + "“")
                    result_in_quotation.append(False)
                buffer = ""
                state = "IN_CHINESE_QUOTES"
            else:
                buffer += char

        elif state == "IN_ENGLISH_QUOTES":
            if char == '"':
                # 找到闭合引号，确认这是一个有效的引号内区域
                result.extend(quote_content_buffer)
                result_in_quotation.extend([True] * len(quote_content_buffer))
                # 清空状态和暂存区
                quote_content_buffer = []
                state = "OUTSIDE"
                buffer = '"'
            else:
                # 未找到闭合引号，将内容暂存
                quote_content_buffer.append(char)

        elif state == "IN_CHINESE_QUOTES":
            if char == "”":
                # 找到闭合引号，确认这是一个有效的引号内区域
                result.extend(quote_content_buffer)
                result_in_quotation.extend([True] * len(quote_content_buffer))
                # 清空状态和暂存区
                quote_content_buffer = []
                state = "OUTSIDE"
                buffer = "”"
            else:
                # 未找到闭合引号，将内容暂存
                quote_content_buffer.append(char)

    # --- 关键的后处理 ---
    # 如果循环结束后仍在引号内状态，说明引号未闭合
    if state != "OUTSIDE":
        # 将未闭合的起始引号、暂存的内容和最后的缓冲区内容全部合并
        malformed_text = "".join(quote_content_buffer) + buffer

        # 与前一个 "引号外" 的部分合并(如果存在)，以获得更干净的输出
        if result and not result_in_quotation[-1]:
            result[-1] += malformed_text
        else:
            result.append(malformed_text)
            result_in_quotation.append(False)
    # 如果最后一部分是正常的引号外文本
    elif buffer:
        result.append(buffer)
        result_in_quotation.append(False)

    return result, result_in_quotation


def split_string_in_quotation_and_special_tokens(
    s: str,
    prefix_len: int,
    special_tokens: list[str],
    should_split_in_quatation=True,
) -> list[str] | tuple[list[str], list[bool]]:
    """
    首先按一个明确的special_tokens列表分割字符串, 然后对非token部分再按引号分割。

    Args:
        s (str): 待分割的完整字符串。
        prefix_len (int): 整个字符串的前缀长度。
        special_tokens (list[str]): 一个包含所有应被视为分隔符的特殊token的列表。
        should_split_in_quatation (bool, optional): 是否应执行引号分割。默认为 True。

    Returns:
        list[str] | tuple[list[str], list[bool]]: 分割结果。
    """
    if not special_tokens:
        if should_split_in_quatation:
            return split_string_in_quotation(s, prefix_len)
        return [s], [False]

    # 1. 按长度降序排序，以优先匹配更长的token(例如，确保`<|im_start|>`优先于`<|im|>`被匹配)。
    escaped_tokens = sorted(set(special_tokens), key=len, reverse=True)

    # 2. 从列表动态构建正则表达式。
    #    用'|'(或)连接所有token，并用'()'包裹形成一个捕获组，
    #    这样 re.split 就会保留匹配到的token。
    pattern = f"({'|'.join(escaped_tokens)})"

    parts = re.split(pattern, s)
    parts = [x for x in parts if len(x) > 0]
    parts_start_index = np.cumsum([0] + [len(part) for part in parts[:-1]]).tolist()

    if not should_split_in_quatation:
        return parts
    else:
        processed_parts = []
        for part, part_start_index in zip(parts, parts_start_index):
            # 使用 `part in special_tokens` 进行精确检查，比 re.match 更可靠
            if part not in special_tokens:
                split_result = split_string_in_quotation(part, max(0, prefix_len - part_start_index))
                processed_parts.append(split_result)
            else:
                processed_parts.append(([part], [False]))

        if not processed_parts:
            return [], []

        split_str_processed_parts, in_quatation_flag_processed_parts = zip(*processed_parts)
        split_str_processed_parts = functools.reduce(operator.iadd, split_str_processed_parts, [])
        in_quatation_flag_processed_parts = functools.reduce(operator.iadd, in_quatation_flag_processed_parts, [])
        return split_str_processed_parts, in_quatation_flag_processed_parts
