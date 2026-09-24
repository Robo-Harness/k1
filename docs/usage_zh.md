# robo-harness k1 使用说明

这是一套让视觉语言模型通过感知、运动和记忆工具控制仿真机器人的框架。仅提供 LIBERO-Pro 与 RoboSuite 接入，算法统一称为 **robo-harness k1**。

## 从哪里开始读

- `src/robo_harness/runtime.py`：模型每轮看什么、有哪些工具、怎样执行及记录。
- `libero_adapter.py`、`robosuite_adapter.py`：环境适配与真实控制器执行。
- `perception.py`、`regions.py`、`tracked_points.py`：测深度、区域几何、分割和跟踪。
- `motion.py`、`position_reach.py`、`pose_targets.py`：相对运动、绝对位置到达、姿态控制。
- `episode_history.py`、`task_progress.py`：完整历史、检索与模型维护的里程碑。
- `training/prepare.py`、`training/sft.py`：用用户自备数据整理监督样本和进行 LoRA 微调。

完整技术说明见 [architecture.md](architecture.md)，依赖安装见 [environments.md](environments.md)。

## 使用顺序

1. 为两个仿真环境分别准备 Python 环境，不要混装控制器接口不同的依赖。
2. 安装本项目，运行单元测试。
3. 按环境文档准备外部源码、资产与初始状态；按需提供 SAM3、TAPNext++ 权重。它们不随本仓库分发。
4. 复制或编辑 `configs/` 中的模板，先运行 `--smoke`，确认复位、渲染和原生控制。
5. 在进程环境变量里配置模型名称、API 地址和密钥，再运行正式模型闭环。
6. 评测输出全部放到忽略目录 `outputs/`。需要发给其他人时另行检查，不要直接加入源码仓库。

N 默认 8，表示最近完整文字交互轮数；K 默认 1，表示额外保留上一轮的图片。完整历史不会因为当前提示截断而被删除。模型可以检索旧证据和更新自己的进度记录。

`delta_axis` 默认 0.03 米，约束相对运动的每个分量，不是总向量长度。`move_toward` 尝试一次调用内到达绝对位置，但内部仍执行多步原生反馈控制，并不瞬移，也不保证有可行路径。

## 开源材料边界

目录只包含源码、模板、文档和合成测试，不含实验数据、模型权重、视频、API key、个人路径或开发日志。运行生成的资料不自动公开。MIT 授权仅适用于本项目代码，第三方资源遵守各自授权。

微调接口保留，数据须用户自行准备。模型的简短 `decision_note` 是可见决策摘要，不是隐藏推理的导出。单元测试通过不等于重新验证了所有仿真任务的成功率；整理后的安装环境仍应先做本地 smoke 与闭环验证。
