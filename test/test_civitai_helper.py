#!/usr/bin/env python3
"""
ComfyUI Civitai Helper 测试脚本

用于验证核心功能是否正常工作
"""

import sys
import os
import json


# 获取当前脚本的目录
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(current_dir))
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))
# sys.path.append(r"D:\APPs\stable-diffusion\ComfyUI-aki-v1.3")
import folder_paths
from py.civitai_helper import CivitaiModelBatchProcessor

def test_batch_processor():
    """测试批量处理器"""
    print("\n=== 测试批量处理器 ===")
    processor = CivitaiModelBatchProcessor()
    
    # 测试批量处理（仅测试逻辑，不实际下载）
    try:
        status, total, successful, failed = processor.batch_process_models(
            model_type="lora",
            skip_existing_info=True,
            download_preview_images=True,
            proxy="socks5://127.0.0.1:7888"
        )
        
        print(f"处理状态: {status}")
        print(f"总处理数量: {total}")
        print(f"成功数量: {successful}")
        print(f"失败数量: {len(failed) if isinstance(failed, list) else 0}")
        
        if isinstance(failed, list) and len(failed) > 0:
            print(f"失败模型示例: {failed[0]}")
            
    except Exception as e:
        print(f"批量处理测试出错: {e}")

def test_input_types():
    """测试节点输入类型定义"""
    print("\n=== 测试节点输入类型 ===")
    
    # 测试 CivitaiModelBatchProcessor 的输入类型
    input_types = CivitaiModelBatchProcessor.INPUT_TYPES()
    print("CivitaiModelBatchProcessor 输入类型:")
    print(json.dumps(input_types, indent=2, ensure_ascii=False))
    
    # 验证必需参数
    required = input_types.get("required", {})
    optional = input_types.get("optional", {})
    
    print(f"必需参数: {list(required.keys())}")
    print(f"可选参数: {list(optional.keys())}")

def main():
    """主测试函数"""
    # path = folder_paths.get_folder_paths("loras")
    # print(path)
    # return
    print("ComfyUI Civitai Helper 测试开始")
    print("=" * 50)
    
    test_input_types()
    test_batch_processor()
    print("\n" + "=" * 50)
    print("测试完成")

if __name__ == "__main__":
    main()