import json
import base64
import io
import requests
import numpy as np
from PIL import Image
import torch
import traceback
from typing import Optional, Tuple, Dict, Any

class ChatCompletionsNode:
    """
    调用 OpenAI 兼容的聊天补全 API 接口
    支持文本和图片输入
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # API 配置
                "api_base_url": ("STRING", {
                    "default": "http://127.0.0.1:5102",
                    "multiline": False
                }),
                "api_endpoint": ("STRING", {
                    "default": "/v1/chat/completions",
                    "multiline": False
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                
                # 模型配置
                "model": ("STRING", {
                    "default": "claude-opus-4-1-20250805",
                    "multiline": False
                }),
                
                # 提示词
                "system_prompt": ("STRING", {
                    "default": "You are a helpful assistant.",
                    "multiline": True
                }),
                "user_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": False
                }),
                
                # 生成参数
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "max_tokens": ("INT", {
                    "default": 4096,
                    "min": 1,
                    "max": 128000,
                    "step": 1,
                    "display": "number"
                }),
                "stream": (["disable", "enable"], {
                    "default": "disable"
                }),
            },
            "optional": {
                # 可选的图片输入
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("response", "error", "prompt_tokens", "completion_tokens")
    FUNCTION = "call_api"
    CATEGORY = "API/OpenAI"
    
    def tensor_to_base64(self, tensor_image):
        """将 ComfyUI 的 tensor 图片转换为 base64 编码"""
        try:
            # ComfyUI 图片格式: [batch, height, width, channels], 值范围 [0, 1]
            if len(tensor_image.shape) == 4:
                tensor_image = tensor_image[0]  # 取第一张图片
            
            # 转换为 numpy 数组并调整值范围到 [0, 255]
            image_np = (tensor_image.cpu().numpy() * 255).astype(np.uint8)
            
            # 转换为 PIL Image
            image = Image.fromarray(image_np, mode='RGB')
            
            # 转换为 base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return f"data:image/png;base64,{img_base64}"
        except Exception as e:
            print(f"图片转换失败: {str(e)}")
            return None
    
    def call_api(self, api_base_url, api_endpoint, api_key, model, system_prompt,
                 user_prompt, temperature, max_tokens, stream, image=None):
        """
        调用 OpenAI API（同步版本）
        """
        error_msg = ""
        response_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        
        try:
            # 构建请求 URL
            url = f"{api_base_url.rstrip('/')}{api_endpoint}"
            
            # 构建消息列表
            messages = []
            
            # 添加系统消息
            if system_prompt.strip():
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # 构建用户消息
            user_message = {"role": "user"}
            
            # 如果有图片输入，构建多模态消息
            if image is not None:
                image_base64 = self.tensor_to_base64(image)
                if image_base64:
                    user_message["content"] = [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_base64}}
                    ]
                else:
                    # 图片转换失败，只使用文本
                    user_message["content"] = user_prompt
                    error_msg = "警告: 图片转换失败，仅使用文本输入"
            else:
                user_message["content"] = user_prompt
            
            messages.append(user_message)
            
            # 构建请求体
            request_body = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream == "enable"
            }
            
            # 构建请求头
            headers = {
                "Content-Type": "application/json"
            }
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            # 发送请求
            print(f"正在调用 API: {url}")
            
            if stream == "enable":
                # 处理流式响应
                response = requests.post(
                    url,
                    json=request_body,
                    headers=headers,
                    stream=True,
                    timeout=300
                )
                
                if response.status_code != 200:
                    error_msg = f"API 错误 ({response.status_code}): {response.text}"
                    print(f"API 调用失败: {error_msg}")
                    return ("", error_msg, 0, 0)
                
                response_text = self._handle_stream_response(response)
            else:
                # 处理非流式响应
                response = requests.post(
                    url,
                    json=request_body,
                    headers=headers,
                    timeout=300
                )
                
                if response.status_code != 200:
                    error_msg = f"API 错误 ({response.status_code}): {response.text}"
                    print(f"API 调用失败: {error_msg}")
                    return ("", error_msg, 0, 0)
                
                response_json = response.json()
                
                # 检查错误
                if "error" in response_json:
                    error_msg = f"API 返回错误: {response_json['error']}"
                    print(error_msg)
                    return ("", error_msg, 0, 0)
                
                # 提取响应文本
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    response_text = response_json["choices"][0]["message"]["content"]
                
                # 提取 token 使用信息
                if "usage" in response_json:
                    prompt_tokens = response_json["usage"].get("prompt_tokens", 0)
                    completion_tokens = response_json["usage"].get("completion_tokens", 0)
            
            return (response_text, error_msg, prompt_tokens, completion_tokens)
            
        except requests.Timeout:
            error_msg = "API 请求超时（300秒）"
            print(f"错误: {error_msg}")
            return ("", error_msg, 0, 0)
        except requests.RequestException as e:
            error_msg = f"网络错误: {str(e)}"
            print(f"错误: {error_msg}")
            return ("", error_msg, 0, 0)
        except Exception as e:
            error_msg = f"未知错误: {str(e)}\n{traceback.format_exc()}"
            print(f"错误: {error_msg}")
            return ("", error_msg, 0, 0)
    
    def _handle_stream_response(self, response):
        """处理流式响应（同步版本）"""
        full_response = ""
        try:
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        data_str = line[6:]  # 移除 "data: " 前缀
                        
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    full_response += delta["content"]
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"处理流式响应时出错: {str(e)}")
        
        return full_response


class ImageGenerationsNode:
    """
    调用 OpenAI 兼容的图像生成 API 接口
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # API 配置
                "api_base_url": ("STRING", {
                    "default": "http://127.0.0.1:5102",
                    "multiline": False
                }),
                "api_endpoint": ("STRING", {
                    "default": "/v1/images/generations",
                    "multiline": False
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                
                # 生成参数
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": False
                }),
                "model": ("STRING", {
                    "default": "imagen-4.0-ultra-generate-preview-06-06",
                    "multiline": False
                }),
                "n": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "number"
                }),
                "size": (["1024x1024", "512x512", "256x256"], {
                    "default": "1024x1024"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "error")
    FUNCTION = "generate_images"
    CATEGORY = "API/OpenAI"
    
    def generate_images(self, api_base_url, api_endpoint, api_key, prompt,
                       model, n, size):
        """
        生成图片（同步版本）
        """
        error_msg = ""
        images_tensor = None
        
        try:
            # 构建请求 URL
            url = f"{api_base_url.rstrip('/')}{api_endpoint}"
            
            # 构建请求体
            request_body = {
                "prompt": prompt,
                "model": model,
                "n": n,
                "size": size
            }
            
            # 构建请求头
            headers = {
                "Content-Type": "application/json"
            }
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            print(f"正在调用图像生成 API: {url}")
            print(f"请求参数: {json.dumps(request_body, ensure_ascii=False)}")
            
            # 发送请求
            response = requests.post(
                url,
                json=request_body,
                headers=headers,
                timeout=300
            )
            
            if response.status_code != 200:
                error_msg = f"API 错误 ({response.status_code}): {response.text}"
                print(f"API 调用失败: {error_msg}")
                # 返回一个空的黑色图片作为占位符
                empty_image = torch.zeros((1, 64, 64, 3))
                return (empty_image, error_msg)
            
            response_json = response.json()
            
            # 检查错误
            if "error" in response_json:
                error_msg = f"API 返回错误: {response_json['error']}"
                print(error_msg)
                empty_image = torch.zeros((1, 64, 64, 3))
                return (empty_image, error_msg)
            
            # 处理返回的图片数据
            if "data" in response_json:
                images = []
                
                for item in response_json["data"]:
                    image_tensor = None
                    
                    if "url" in item:
                        # 处理 URL 格式的图片
                        image_url = item["url"]
                        print(f"获取图片 URL: {image_url}")
                        
                        # 下载图片
                        image_tensor = self._download_image(image_url)
                        
                    elif "b64_json" in item:
                        # 处理 base64 格式的图片
                        base64_data = item["b64_json"]
                        image_tensor = self._base64_to_tensor(base64_data)
                    
                    if image_tensor is not None:
                        images.append(image_tensor)
                
                if images:
                    # 将所有图片堆叠成一个批次
                    images_tensor = torch.cat(images, dim=0)
                    print(f"成功生成 {len(images)} 张图片")
                else:
                    error_msg = "未能从 API 响应中提取任何图片"
                    print(error_msg)
                    empty_image = torch.zeros((1, 64, 64, 3))
                    return (empty_image, error_msg)
            else:
                error_msg = "API 响应格式错误：缺少 'data' 字段"
                print(error_msg)
                empty_image = torch.zeros((1, 64, 64, 3))
                return (empty_image, error_msg)
            
            return (images_tensor, error_msg)
            
        except requests.Timeout:
            error_msg = "API 请求超时（300秒）"
            print(f"错误: {error_msg}")
            empty_image = torch.zeros((1, 64, 64, 3))
            return (empty_image, error_msg)
        except requests.RequestException as e:
            error_msg = f"网络错误: {str(e)}"
            print(f"错误: {error_msg}")
            empty_image = torch.zeros((1, 64, 64, 3))
            return (empty_image, error_msg)
        except Exception as e:
            error_msg = f"未知错误: {str(e)}\n{traceback.format_exc()}"
            print(f"错误: {error_msg}")
            empty_image = torch.zeros((1, 64, 64, 3))
            return (empty_image, error_msg)
    
    def _download_image(self, url):
        """从 URL 下载图片并转换为 tensor（同步版本）"""
        try:
            # 如果是 data URL，直接解析
            if url.startswith("data:"):
                # 提取 base64 数据
                base64_data = url.split(",")[1] if "," in url else url
                return self._base64_to_tensor(base64_data)
            
            # 否则从网络下载
            response = requests.get(url, timeout=60)
            if response.status_code != 200:
                print(f"下载图片失败: HTTP {response.status_code}")
                return None
            
            image_data = response.content
            
            # 转换为 PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # 确保是 RGB 格式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 转换为 numpy 数组
            image_np = np.array(image).astype(np.float32) / 255.0
            
            # 转换为 tensor [1, height, width, channels]
            image_tensor = torch.from_numpy(image_np).unsqueeze(0)
            
            return image_tensor
            
        except Exception as e:
            print(f"下载图片时出错: {str(e)}")
            return None
    
    def _base64_to_tensor(self, base64_data):
        """将 base64 数据转换为 tensor"""
        try:
            # 解码 base64
            image_data = base64.b64decode(base64_data)
            
            # 转换为 PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # 确保是 RGB 格式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 转换为 numpy 数组
            image_np = np.array(image).astype(np.float32) / 255.0
            
            # 转换为 tensor [1, height, width, channels]
            image_tensor = torch.from_numpy(image_np).unsqueeze(0)
            
            return image_tensor
            
        except Exception as e:
            print(f"转换 base64 图片时出错: {str(e)}")
            return None

# 导出节点类
NODE_CLASS_MAPPINGS = {
    "ChatCompletionsNode": ChatCompletionsNode,
    "ImageGenerationsNode": ImageGenerationsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatCompletionsNode": "OpenAI 聊天补全",
    "ImageGenerationsNode": "OpenAI 图像生成"
}