- kit = ComfyKit(comfyui_url="...") （ comfykit_demo.py:L5 ）这一步 不是 创建网络 session，只是创建一个 SDK 对象并保存配置（地址、执行器类型等）。
- result = await kit.execute(...) （ comfykit_demo.py:L6 ）这一步才真正开始执行流程：读 workflow、提交任务、等待结果。
默认是 HTTP 执行器（ executor_type="http" ），所以行为是：
- 先 POST /prompt 提交任务（ http_executor.py:L32-L39 ）
- 再轮询 GET /history/{prompt_id} 等结果（ http_executor.py:L73-L80 ）
- 这些请求的 session 在内部通过 aiohttp.ClientSession 临时创建（ base_executor.py:L85-L99 ）