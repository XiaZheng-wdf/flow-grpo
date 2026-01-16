# config/pickscore_gdpo.py
import os
import imp
import ml_collections

# 1. 动态加载同目录下的 grpo.py 模块
# 这样我们可以直接调用其中已经写好的配置函数
grpo = imp.load_source("grpo", os.path.join(os.path.dirname(__file__), "grpo.py"))


def get_config():
    # 2. 获取基础配置：完全复用 grpo.py 中 pickscore_sd3 的所有参数
    # 这包括了模型路径(SD3.5)、采样步数、Batch Size计算、KL惩罚系数(beta)等所有关键设置
    config = grpo.pickscore_sd3()

    # 3. GDPO 专属修改：覆盖奖励函数设置
    # 这里我们定义两个奖励，GDPO 会自动识别并分别进行归一化
    config.reward_fn = {
        "pickscore": 1.0,  # 主任务：图文一致性
        "aesthetic": 0.5,  # 辅助任务：美学质量 (权重0.5表示相对重要性)
    }

    # 4. 修改实验名称和保存路径，避免覆盖原有的 GRPO 实验日志
    config.run_name = "pickscore_aesthetic_gdpo"
    config.save_dir = 'logs/pickscore/sd3.5-M-gdpo'

    return config