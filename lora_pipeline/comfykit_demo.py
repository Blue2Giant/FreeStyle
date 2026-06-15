import asyncio
from comfykit import ComfyKit

async def main():
    async with ComfyKit(comfyui_url="http://10.201.19.23:8188",session_pool_size=2) as kit:
        result = await kit.execute("/data/LoraPipeline/assets/SDXL_illustrious_magic.json")
        print(result.images)  # ['http://10.201.16.49:8188/view?filename=...']

if __name__ == "__main__":
    asyncio.run(main())