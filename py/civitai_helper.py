"""
ComfyUI Civitai Helper

借鉴 Stable-Diffusion-Webui-Civitai-Helper 的模型元数据、封面图下载功能，适配 ComfyUI 架构和 API。

作者: Kilo Code
版本: 1.0.0
"""

import os
import json
import hashlib
import requests
import urllib3
from typing import Dict, List, Optional, Tuple, Any
import folder_paths
import time
import logging

# 禁用 SSL 警告
urllib3.disable_warnings()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# 基础设施层 (Infrastructure Layer)
# ============================================================================

class CivitaiAPIClient:
    """Civitai API 客户端"""

    BASE_URL = "https://civitai.com/api/v1"
    TIMEOUT = 5  # 5秒超时
    MAX_RETRIES = 1
    RETRY_DELAY = 1.0

    def __init__(self, proxy: Optional[str] = None):
        self.headers = {
            "User-Agent": "ComfyUI-CivitaiHelper/1.0.0",
            "Accept": "application/json",
        }
        self.proxy = proxy

    def _request(self, url: str, method: str = "GET", **kwargs) -> Optional[Dict]:
        """通用请求方法"""
        # 设置代理
        proxies = {"https": self.proxy, "http": self.proxy} if self.proxy else None
        
        last_exception = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.request(
                    method,
                    url,
                    headers=self.headers,
                    timeout=self.TIMEOUT,
                    verify=False,
                    proxies=proxies,
                    **kwargs
                )

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    logger.warning(f"Resource not found: {url}")
                    return None
                elif response.status_code == 429:
                    # Rate limited, wait and retry
                    wait_time = 2 ** attempt
                    logger.info(f"Rate limited, waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"API request failed: {response.status_code} - {response.reason}")
                    return None

            except requests.exceptions.Timeout as e:
                last_exception = e
                logger.warning(f"Request timeout for {url}, attempt {attempt + 1}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY)
                continue
            except Exception as e:
                last_exception = e
                logger.error(f"Request error: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY)
                continue

        # 所有重试都失败了，抛出最后一个异常
        if last_exception:
            raise last_exception
        else:
            raise Exception("All retry attempts failed")

    def get_model_by_hash(self, model_hash: str) -> Optional[Dict]:
        """通过 SHA256 获取模型信息"""
        if not model_hash:
            return None

        url = f"{self.BASE_URL}/model-versions/by-hash/{model_hash}"
        logger.info(f"Fetching model info by hash: {model_hash}")
        return self._request(url)

    def get_model_by_id(self, model_id: str) -> Optional[Dict]:
        """通过模型 ID 获取模型信息"""
        if not model_id:
            return None

        url = f"{self.BASE_URL}/models/{model_id}"
        logger.info(f"Fetching model info by ID: {model_id}")
        return self._request(url)

    def get_model_version_by_id(self, version_id: str) -> Optional[Dict]:
        """通过版本 ID 获取模型版本信息"""
        if not version_id:
            return None

        url = f"{self.BASE_URL}/model-versions/{version_id}"
        logger.info(f"Fetching model version by ID: {version_id}")
        return self._request(url)

    def download_image(self, image_url: str) -> Optional[bytes]:
        """下载图片数据"""
        try:
            # 设置代理
            proxies = {"https": self.proxy, "http": self.proxy} if self.proxy else None
            
            response = requests.get(
                image_url, 
                stream=True,
                verify=False, 
                timeout=self.TIMEOUT,
                proxies=proxies,
            )
            if response.status_code == 200:
                return response.content
            else:
                logger.error(f"Failed to download image: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error downloading image: {e}")
            return None


class FileUtils:
    """文件操作工具类"""

    @staticmethod
    def calculate_sha256(file_path: str, chunk_size: int = 8192) -> str:
        """计算文件的 SHA256 哈希值"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)

        return sha256.hexdigest()

    @staticmethod
    def calculate_sha256_with_progress(file_path: str, progress_callback=None) -> str:
        """带进度回调的 SHA256 计算"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        sha256 = hashlib.sha256()
        file_size = os.path.getsize(file_path)
        processed = 0

        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
                processed += len(chunk)
                if progress_callback:
                    progress = processed / file_size
                    progress_callback(progress, f"Hashing: {progress:.1%}")

        return sha256.hexdigest()

    @staticmethod
    def get_file_info(file_path: str) -> Dict[str, Any]:
        """获取文件信息"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        stat = os.stat(file_path)
        return {
            "path": file_path,
            "name": os.path.basename(file_path),
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "extension": os.path.splitext(file_path)[1].lower()
        }

    @staticmethod
    def ensure_directory(path: str) -> None:
        """确保目录存在"""
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def is_valid_model_file(file_path: str) -> bool:
        """检查是否为有效的模型文件"""
        if not os.path.exists(file_path):
            return False

        valid_extensions = folder_paths.supported_pt_extensions
        return os.path.splitext(file_path)[1].lower() in valid_extensions


class PathResolver:
    """路径解析器"""

    MODEL_TYPES = {
        "checkpoint": "checkpoints",
        "lora": "loras",
        "vae": "vae",
        "embeddings": "embeddings",
        "controlnet": "controlnet",
        "upscale": "upscale_models"
    }

    def __init__(self):
        self.base_paths = self._get_base_paths()

    def _get_base_paths(self) -> Dict[str, list[str]]:
        """获取基础路径"""
        paths = {}
        for model_type, folder_name in self.MODEL_TYPES.items():
            try:
                paths[model_type] = folder_paths.get_folder_paths(folder_name)
            except:
                # 如果获取失败，使用默认路径
                paths[model_type] = os.path.join("models", folder_name)

        return paths

    def get_model_directory(self, model_type: str) -> list[str]:
        """获取模型目录"""
        return self.base_paths.get(model_type, [f"models/{model_type}"])

    def resolve_model_path(self, model_type: str, model_name: str) -> Optional[str]:
        """解析模型路径"""
        base_dirs = self.get_model_directory(model_type)

        for base_dir in base_dirs:
            # 尝试直接路径
            full_path = os.path.join(base_dir, model_name)
            if os.path.exists(full_path):
                return full_path

            # 尝试在子目录中查找
            for root, dirs, files in os.walk(base_dir):
                if model_name in files:
                    return os.path.join(root, model_name)

        return None

    def get_metadata_path(self, model_path: str) -> str:
        """获取元数据文件路径"""
        base, ext = os.path.splitext(model_path)
        return f"{base}.civitai.json"

    def get_preview_path(self, model_path: str) -> str:
        """获取预览图片路径"""
        base, ext = os.path.splitext(model_path)
        return f"{base}.preview.png"


# ============================================================================
# 服务层 (Service Layer)
# ============================================================================

class MetadataService:
    """元数据服务"""

    def __init__(self, api_client: CivitaiAPIClient, path_resolver: PathResolver):
        self.api_client = api_client
        self.path_resolver = path_resolver

    def fetch_model_metadata(self, model_hash: str) -> Tuple[Optional[Dict], bool]:
        """获取模型元数据
        返回: (metadata, is_network_error)
        - metadata: 元数据字典或None
        - is_network_error: 是否为网络错误
        """
        try:
            data = self.api_client.get_model_by_hash(model_hash)
            if data:
                # 获取父模型信息
                return self._enrich_metadata(data), False
            else:
                # API返回None，表示模型不存在（404）
                return None, False
        except Exception as e:
            # 网络错误或其他异常
            logger.error(f"Network error fetching metadata: {e}")
            return None, True

    def _enrich_metadata(self, data: Dict) -> Dict:
        """丰富元数据信息"""
        try:
            model_id = data.get("modelId")
            if model_id:
                parent_info = self.api_client.get_model_by_id(model_id)
                if parent_info:
                    # 合并父模型信息
                    data["model"] = parent_info
        except Exception as e:
            logger.warning(f"Failed to enrich metadata: {e}")

        return data

    def save_metadata(self, model_path: str, metadata: Dict) -> None:
        """保存元数据到文件"""
        metadata_path = self.path_resolver.get_metadata_path(model_path)

        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Metadata saved to: {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def load_metadata(self, model_path: str) -> Optional[Dict]:
        """从文件加载元数据"""
        metadata_path = self.path_resolver.get_metadata_path(model_path)

        if not os.path.exists(metadata_path):
            return None

        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return None


class ImageService:
    """图片服务"""

    def __init__(self, path_resolver: PathResolver):
        self.path_resolver = path_resolver
        self.api_client = None  # 将在 CivitaiImageDownloader 中注入

    def download_preview_image(self, image_url: str, model_path: str,
                              progress_callback=None) -> Optional[str]:
        """下载预览图片"""
        preview_path = self.path_resolver.get_preview_path(model_path)

        try:
            # 使用 API 客户端下载图片
            image_data = self.api_client.download_image(image_url)
            if image_data is None:
                return None

            # 保存图片数据到文件
            with open(preview_path, 'wb') as f:
                f.write(image_data)

            logger.info(f"Preview image saved to: {preview_path}")
            return preview_path

        except Exception as e:
            logger.error(f"Failed to download preview image: {e}")
            return None

    def has_preview_image(self, model_path: str) -> bool:
        """检查是否存在预览图片"""
        preview_path = self.path_resolver.get_preview_path(model_path)
        return os.path.exists(preview_path)


class ScannerService:
    """扫描服务"""

    def __init__(self, path_resolver: PathResolver, file_utils: FileUtils):
        self.path_resolver = path_resolver
        self.file_utils = file_utils

    def scan_models(self, model_type: str, skip_existing_info: bool = True, progress_callback=None) -> List[Dict]:
        """扫描指定类型的模型文件"""
        base_dirs = self.path_resolver.get_model_directory(model_type)
        logger.info(f"开始扫描模型目录：{base_dirs}")
        models = []

        valid_base_dirs = []
        for base_dir in base_dirs:
            if not os.path.exists(base_dir):
                logger.warning(f"Model directory does not exist: {base_dir}")
            else:
                valid_base_dirs.append(base_dir)
        if len(valid_base_dirs) == 0:
            logger.error("No valid model directories found.")
            return []
        base_dirs = valid_base_dirs

        for base_dir in base_dirs:
            # 先统计需要处理的文件数量
            logger.info(f"扫描目录：{base_dir}")
            total_files = 0
            for root, dirs, files in os.walk(base_dir):
                for f in files:
                    file_path = os.path.join(root, f)
                    if self.file_utils.is_valid_model_file(file_path):
                        if skip_existing_info:
                            # 检查是否已有 .civitai.info 文件
                            base, ext = os.path.splitext(file_path)
                            info_path = f"{base}.civitai.info"
                            if os.path.exists(info_path):
                                continue
                        total_files += 1
            logger.info(f"需要处理的文件数量：{total_files}")
            processed = 0
            for root, dirs, files in os.walk(base_dir):
                for filename in files:
                    file_path = os.path.join(root, filename)

                    if not self.file_utils.is_valid_model_file(file_path):
                        continue

                    if skip_existing_info:
                        # 检查是否已有 .civitai.info 文件
                        base, ext = os.path.splitext(file_path)
                        info_path = f"{base}.civitai.info"
                        if os.path.exists(info_path):
                            logger.info(f"跳过已有信息文件的模型: {filename}")
                            continue

                    try:
                        # 计算哈希值
                        # file_hash = "0"
                        file_hash = self.file_utils.calculate_sha256_with_progress(
                            file_path,
                            lambda p, msg: progress_callback(p, msg) if progress_callback else None
                        )

                        model_info = {
                            "path": file_path,
                            "name": filename,
                            "hash": file_hash,
                            "size": os.path.getsize(file_path),
                            "modified": os.path.getmtime(file_path),
                            "type": model_type
                        }

                        models.append(model_info)
                        processed += 1

                        if progress_callback:
                            overall_progress = processed / total_files if total_files > 0 else 1.0
                            progress_callback(overall_progress, f"Scanned: {filename}")

                    except Exception as e:
                        logger.error(f"Failed to process {filename}: {e}")
                        continue

        return models

# ============================================================================
# ComfyUI 节点 (ComfyUI Nodes)
# ============================================================================

class CivitaiModelBatchProcessor:
    """Civitai 模型批量处理器 - 集成扫描、元数据获取和图片下载功能"""

    def __init__(self):
        self.api_client = CivitaiAPIClient()
        self.path_resolver = PathResolver()
        self.file_utils = FileUtils()
        self.metadata_service = MetadataService(self.api_client, self.path_resolver)
        self.image_service = ImageService(self.path_resolver)
        # 注入 API 客户端到 ImageService
        self.image_service.api_client = self.api_client
        self.scanner_service = ScannerService(self.path_resolver, self.file_utils)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["checkpoint", "lora", "vae", "embeddings", "controlnet", "upscale"], {
                    "default": "lora",
                }),
            },
            "optional": {
                "proxy": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "网络代理地址，例如: socks5://127.0.0.1:788"
                }),
                "download_preview_images": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "下载预览图片"
                }),
                "skip_existing_info": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "跳过已有 .civitai.info 文件的模型"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "LIST")
    RETURN_NAMES = ("status", "total_processed", "successful_count", "failed_models")
    FUNCTION = "batch_process_models"
    CATEGORY = "tools/civitai"

    def batch_process_models(self, model_type: str, skip_existing_info: bool = True,
                           download_preview_images: bool = True, proxy: str = ""):
        """批量处理模型：扫描、获取元数据、下载图片"""
        try:
            logger.info(f"开始批量处理 {model_type} 模型")
            
            # 更新 API 客户端的代理设置
            self.api_client.proxy = proxy if proxy else None
            
            # 1. 扫描模型文件（已经在扫描阶段过滤了已有信息文件的模型）
            models = self.scanner_service.scan_models(model_type, skip_existing_info)
            if not models:
                return ("WARNING: 未找到任何需要处理的模型文件", 0, 0, [])

            logger.info(f"共找到 {len(models)} 个需要处理的模型文件")

            # 2. 批量处理模型
            successful_count = 0
            failed_models = []
            total_processed = len(models)

            for i, model in enumerate(models):
                try:
                    model_path = model["path"]
                    model_name = model["name"]
                    model_hash = model["hash"]

                    logger.info(f"处理模型 ({i+1}/{total_processed}): {model_name}")
                    # continue # 取消注释后不执行元数据、图片下载

                    # 获取元数据
                    metadata = self._fetch_and_save_metadata(model_path, model_hash)
                    
                    if metadata and download_preview_images:
                        # 下载预览图片
                        self._download_preview_images(model_path, metadata)

                    successful_count += 1
                    logger.info(f"成功处理模型: {model_name}")

                except Exception as e:
                    error_msg = f"处理模型失败 {model['name']}: {str(e)}"
                    logger.error(error_msg)
                    failed_models.append({
                        "name": model["name"],
                        "path": model["path"],
                        "error": str(e)
                    })

                # 添加延迟避免API限制
                time.sleep(0.5)

            # 4. 生成结果报告
            if successful_count == total_processed:
                status = f"SUCCESS: 成功处理 {successful_count}/{total_processed} 个模型"
            elif successful_count > 0:
                status = f"PARTIAL: 成功处理 {successful_count}/{total_processed} 个模型，{len(failed_models)} 个失败"
            else:
                status = f"ERROR: 所有 {total_processed} 个模型处理失败"

            return (status, total_processed, successful_count, failed_models)

        except Exception as e:
            logger.error(f"批量处理错误: {e}")
            return (f"ERROR: {str(e)}", 0, 0, [])

    def _get_civitai_info_path(self, model_path: str) -> str:
        """获取 .civitai.info 文件路径"""
        base, ext = os.path.splitext(model_path)
        return f"{base}.civitai.info"

    def _fetch_and_save_metadata(self, model_path: str, model_hash: str) -> Optional[Dict]:
        """获取并保存元数据"""
        try:
            # 从 Civitai 获取元数据
            metadata, is_network_error = self.metadata_service.fetch_model_metadata(model_hash)
            
            if metadata:
                # 保存到 .civitai.info 文件
                info_path = self._get_civitai_info_path(model_path)
                with open(info_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                logger.info(f"元数据已保存到: {info_path}")
                return metadata
            elif not is_network_error:
                # 当 Civitai 明确返回模型不存在时（非网络错误），创建占位的 .civitai.info 文件
                logger.warning(f"未找到模型元数据，创建占位文件: {os.path.basename(model_path)}")
                self._create_placeholder_metadata(model_path, model_hash)
                return None
            else:
                # 网络错误时不创建占位文件
                logger.warning(f"网络错误，跳过创建占位文件: {os.path.basename(model_path)}")
                return None

        except Exception as e:
            logger.error(f"获取元数据失败: {e}")
            return None

    def _create_placeholder_metadata(self, model_path: str, model_hash: str) -> None:
        """创建占位的元数据文件"""
        try:
            # 创建基础的占位元数据结构
            filename = os.path.basename(model_path)
            base_name = os.path.splitext(filename)[0]
            file_size = os.path.getsize(model_path) // 1024  # KB
            
            placeholder_metadata = {
                "id": "",
                "modelId": "",
                "name": base_name,
                "trainedWords": [],
                "baseModel": "Unknown",
                "description": "",
                "model": {
                    "name": base_name,
                    "type": "",
                    "nsfw": False,
                    "poi": False
                },
                "files": [
                    {
                        "name": filename,
                        "sizeKB": file_size,
                        "type": "Model",
                        "hashes": {
                            "AutoV2": model_hash[:10] if model_hash else "",
                            "SHA256": model_hash if model_hash else ""
                        }
                    }
                ],
                "images": [],
                "tags": [],
                "downloadUrl": "",
                "skeleton_file": True
            }
            
            info_path = self._get_civitai_info_path(model_path)
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(placeholder_metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"占位元数据文件已创建: {info_path}")
            
        except Exception as e:
            logger.error(f"创建占位元数据文件失败: {e}")

    def _download_preview_images(self, model_path: str, metadata: Dict) -> None:
        """下载预览图片"""
        try:
            # 获取预览图片URL
            images = metadata.get("images", [])
            if not images:
                logger.info(f"模型无预览图片: {os.path.basename(model_path)}")
                return

            # 选择第一张图片作为预览
            first_image = images[0]
            image_url = first_image.get("url")
            
            if not image_url:
                logger.warning(f"无效的图片URL: {os.path.basename(model_path)}")
                return

            # 生成预览图片路径 - 使用 .preview.png 格式
            base, ext = os.path.splitext(model_path)
            preview_path = f"{base}.preview.png"

            # 检查是否已存在预览图片（检查多种格式）
            existing_previews = [
                f"{base}.png",
                f"{base}.preview.png",
                f"{base}.jpg",
                f"{base}.preview.jpg"
            ]
            
            for existing_path in existing_previews:
                if os.path.exists(existing_path):
                    logger.info(f"预览图片已存在: {existing_path}")
                    return

            # 下载图片
            logger.info(f"开始为模型 {base}{ext} 下载预览图片: {image_url}")
            try:
                # 使用 API 客户端下载图片数据
                image_data = self.api_client.download_image(image_url)
                if image_data is None:
                    return

                # 保存图片数据到文件
                with open(preview_path, 'wb') as f:
                    f.write(image_data)

                logger.info(f"预览图片已下载: {preview_path}")
                
            except Exception as e:
                logger.error(f"下载预览图片时出错: {e}")

        except Exception as e:
            logger.error(f"下载预览图片失败: {e}")


# ============================================================================
# 工具函数 (Utility Functions)
# ============================================================================

def get_model_hash_from_metadata(metadata: Dict) -> Optional[str]:
    """从元数据中提取模型哈希"""
    try:
        return metadata.get("hash", None)
    except:
        return None


def get_model_download_url(metadata: Dict) -> Optional[str]:
    """从元数据中提取下载链接"""
    try:
        files = metadata.get("files", [])
        if files:
            return files[0].get("downloadUrl", None)
        return None
    except:
        return None


def get_model_preview_urls(metadata: Dict) -> List[str]:
    """从元数据中提取预览图片链接"""
    try:
        images = metadata.get("images", [])
        return [img.get("url", "") for img in images if img.get("url")]
    except:
        return []


def format_metadata_for_display(metadata: Dict) -> str:
    """格式化元数据用于显示"""
    try:
        model_info = metadata.get("model", {})
        version_info = metadata

        info_lines = [
            f"模型名称: {model_info.get('name', 'Unknown')}",
            f"版本名称: {version_info.get('name', 'Unknown')}",
            f"模型ID: {model_info.get('id', 'Unknown')}",
            f"版本ID: {version_info.get('id', 'Unknown')}",
            f"基础模型: {version_info.get('baseModel', 'Unknown')}",
            f"描述: {model_info.get('description', 'No description')[:200]}..."
        ]

        return "\n".join(info_lines)
    except:
        return "无法解析元数据"


# ============================================================================
# 版本信息 (Version Info)
# ============================================================================

__version__ = "1.0.0"
__author__ = "Kilo Code"
__description__ = "ComfyUI Civitai Helper - 完整的 Civitai 模型管理工具"

logger.info(f"{__description__} v{__version__} loaded successfully")