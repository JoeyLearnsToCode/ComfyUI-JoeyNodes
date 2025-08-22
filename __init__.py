from .py.text_nodes import *
from .py.openai_api_nodes import ChatCompletionsNode, ImageGenerationsNode

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "RemoveCommentedText": RemoveCommentedText,
    "ChatCompletionsNode": ChatCompletionsNode,
    "ImageGenerationsNode": ImageGenerationsNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoveCommentedText": "删除注释",
    "ChatCompletionsNode": "OpenAI 聊天补全",
    "ImageGenerationsNode": "OpenAI 图像生成"
}
