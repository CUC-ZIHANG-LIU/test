from diffusers import StableDiffusionPipeline

# 加载不完整的模型
pipeline = StableDiffusionPipeline.from_pretrained("/home/qiqiutang/qqt/AUDIT_my/checkpoint/model_checkpoints/pipeline", local_files_only=True)

# 保存模型，自动生成缺失的文件
pipeline.save_pretrained("/home/qiqiutang/qqt/AUDIT_my/checkpoint/model_checkpoints/pipeline")