supported_pt_extensions: set[str] = {'.ckpt', '.pt', '.pt2', '.bin', '.pth', '.safetensors', '.pkl', '.sft'}
def get_folder_paths(folder_name: str) -> list[str]:
    """ 仅用于测试，返回固定的 lora 模型路径 """
    return ['D:\\APPs\\stable-diffusion\\ComfyUI-aki-v1.3\\models\\checkpoints', 'D:\\APPs\\stable-diffusion\\sd-webui-aki-v4.4\\models\\Stable-diffusion', 'D:\\APPs\\stable-diffusion\\ComfyUI-aki-v1.3\\output\\checkpoints']
    # return ["D:\APPs\stable-diffusion\sd-webui-aki-v4.4\models\Lora"]