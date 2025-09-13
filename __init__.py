from .py.text_nodes import *
from .py.openai_api_nodes import *
from .py.civitai_helper import *

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
WEB_DIRECTORY = "./js"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "RemoveCommentedText": RemoveCommentedText,
    "ChatCompletionsNode": ChatCompletionsNode,
    "ImageGenerationsNode": ImageGenerationsNode,
    # Civitai Helper 节点
    "CivitaiModelBatchProcessor": CivitaiModelBatchProcessor,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoveCommentedText": "删除注释",
    "ChatCompletionsNode": "OpenAI 聊天补全",
    "ImageGenerationsNode": "OpenAI 图像生成",
    "CivitaiModelBatchProcessor": "Civitai 模型批量处理器",
}
