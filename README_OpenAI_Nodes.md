# ComfyUI OpenAI API 节点使用说明

## 概述

本插件为 ComfyUI 添加了两个用于调用 OpenAI 兼容 API 的节点：
- **OpenAI 聊天补全** (ChatCompletionsNode)：调用 `/v1/chat/completions` 接口
- **OpenAI 图像生成** (ImageGenerationsNode)：调用 `/v1/images/generations` 接口

这两个节点专门设计用于与 LMArenaBridge API 服务器配合使用。

## 安装依赖

确保已安装以下 Python 包：
```bash
pip install aiohttp pillow numpy torch
```

## 节点说明

### 1. OpenAI 聊天补全 (ChatCompletionsNode)

**功能**：调用聊天补全 API，支持文本和图片输入。

**输入参数**：
- **api_base_url** (STRING)：API 服务器地址，默认 `http://127.0.0.1:5102`
- **api_endpoint** (STRING)：API 端点路径，默认 `/v1/chat/completions`
- **api_key** (STRING)：API 密钥（可选）
- **model** (STRING)：模型名称，默认 `claude-3-5-sonnet-20241022`
- **system_prompt** (STRING)：系统提示词
- **user_prompt** (STRING)：用户提示词
- **temperature** (FLOAT)：温度参数，范围 0-2，默认 0.7
- **max_tokens** (INT)：最大 token 数，默认 4096
- **stream** (选择框)：是否启用流式响应，选项为 `enable` 或 `disable`
- **image** (IMAGE，可选)：可选的图片输入连接

**输出**：
- **response** (STRING)：AI 响应文本
- **error** (STRING)：错误信息（如果有）
- **prompt_tokens** (INT)：提示词使用的 token 数
- **completion_tokens** (INT)：生成使用的 token 数

**使用示例**：
1. 纯文本对话：只连接文本输入
2. 多模态对话：连接图片输入，可以让 AI 分析图片内容

### 2. OpenAI 图像生成 (ImageGenerationsNode)

**功能**：根据文本提示生成图片。

**输入参数**：
- **api_base_url** (STRING)：API 服务器地址，默认 `http://127.0.0.1:5102`
- **api_endpoint** (STRING)：API 端点路径，默认 `/v1/images/generations`
- **api_key** (STRING)：API 密钥（可选）
- **prompt** (STRING)：图像生成提示词
- **model** (STRING)：模型名称，默认 `dall-e-3`
- **n** (INT)：生成图片数量，范围 1-10，默认 1
- **size** (选择框)：图片尺寸，可选 `1024x1024`、`512x512`、`256x256`

**输出**：
- **images** (IMAGE)：生成的图片（ComfyUI IMAGE 格式）
- **error** (STRING)：错误信息（如果有）

## 使用注意事项

### 1. API 服务器配置
- 确保 LMArenaBridge API 服务器正在运行
- 默认地址是 `http://127.0.0.1:5102`
- 如果服务器在其他地址，请修改 `api_base_url` 参数

### 2. API 密钥
- 如果服务器配置了 API 密钥，需要在 `api_key` 参数中填写
- 如果没有配置密钥，留空即可

### 3. 错误处理
- 节点会捕获并显示所有 API 错误
- 网络超时设置为 300 秒
- 如果 API 调用失败，会在 `error` 输出中显示详细错误信息

### 4. 图片处理
- ChatCompletionsNode 的图片输入会自动转换为 base64 格式
- ImageGenerationsNode 会自动处理返回的 URL 或 base64 图片
- 生成的图片会自动转换为 ComfyUI 的 IMAGE 格式

## 工作流示例

### 示例 1：文本聊天
```
[文本输入] -> [OpenAI 聊天补全] -> [文本显示]
```

### 示例 2：图片分析
```
[加载图片] -> [OpenAI 聊天补全] -> [文本显示]
                        ↑
                  [文本输入："描述这张图片"]
```

### 示例 3：图像生成
```
[文本输入："一只可爱的猫咪"] -> [OpenAI 图像生成] -> [保存图片]
```

## 故障排查

### 常见问题

1. **"API 错误 (401)"**
   - 检查 API 密钥是否正确
   - 确认服务器是否配置了密钥验证

2. **"网络错误"**
   - 检查 API 服务器是否运行
   - 确认 API 地址是否正确
   - 检查防火墙设置

3. **"API 请求超时"**
   - 检查网络连接
   - 可能是模型响应时间过长，请耐心等待

4. **图片转换失败**
   - 确保输入的是有效的 ComfyUI IMAGE 格式
   - 检查图片是否损坏

## 开发说明

### 代码结构
```
ComfyUI-JoeyNodes/
├── __init__.py              # 节点注册
├── py/
│   ├── openai_api_nodes.py  # OpenAI API 节点实现
│   └── text_nodes.py        # 文本处理节点
```

### 扩展建议
- 可以添加更多 API 参数支持（如 top_p、frequency_penalty 等）
- 可以实现对话历史管理
- 可以添加批量处理功能

## 版本历史

### v1.0.0 (2024-01-22)
- 初始版本
- 实现 ChatCompletionsNode
- 实现 ImageGenerationsNode
- 支持流式和非流式响应
- 完善的错误处理机制