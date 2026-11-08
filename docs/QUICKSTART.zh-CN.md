# 中文快速开始

这个库完成 **LAFAN → Primebot Q1 的 FRAX 重定向、播放、碰撞检查与数据验收**。
Q1 是 22 个驱动关节。播放的是运动学参考，不是神经网络策略，也不能证明实机可执行。

## 1. 独立安装

```bash
git clone https://github.com/ChRis98Wang/frax-q1.git
cd frax-q1
python3.12 -m venv .venv
.venv/bin/python -m pip install -c requirements-lock.txt -e '.[retarget,visualization,test]'
export PATH="$PWD/.venv/bin:$PATH"
export Q1_FRAX_DATA="$PWD/.local-data"
q1-frax-setup --with-lafan
```

上游代码和资源均锁定提交。默认缓存为 `~/.cache/q1-frax`，可通过
`Q1_FRAX_DATA` 或 setup 的 `--data-dir` 指定。不要把它指向正在训练的旧项目资源目录。
只播放已有数据时安装 `.[visualization]` 即可，无需 JAX、FRAX、PyTorch 或 Isaac Lab。

GPU 重定向另加 `cuda13` 安装选项，命令显式使用 `--platform cuda`。
`--devices` 只用于 CPU 虚拟设备，不能当作 GPU 数量。GPU 不可用时程序会报错，
不会偷偷改成 CPU 后仍报告 GPU 速度。CPU/GPU 实测条件和结果见仓库首页。

LAFAN 和机器人模型不随本仓库发布。先阅读 [上游许可说明](../THIRD_PARTY_NOTICES.md)。
本地生成的动作、图片和视频默认不会进入 Git 发布内容。

## 2. 重定向与检查

```bash
# 初次验证用 40 秒数据；前几秒包含校准动作，不能当成有效走路。
q1-frax-retarget --pattern 'walk2_subject3.bvh' --max-frames 1200 \
  --devices 1 --out outputs/walk-smoke

# 原始动作可直接播放，但会明确标为 RAW。
q1-frax-play outputs/walk-smoke/npz/walk2_subject3.npz --show-link-labels

# 全帧物理碰撞报告；不打开窗口。
q1-frax-play outputs/walk-smoke/npz/walk2_subject3.npz --check-only \
  --collision-report artifacts/walk-collisions.json

# 截图与视频，保留参考动作及质量状态标记。
q1-frax-play outputs/walk-smoke/npz/walk2_subject3.npz \
  --start-frame 600 --snapshot artifacts/walk.png --show-link-labels
q1-frax-render outputs/walk-smoke/npz/walk2_subject3.npz \
  artifacts/walk.mp4 --start-frame 300 --seconds 10
```

交互按键：

| 按键 | 功能 |
|---|---|
| P / 空格 | 暂停、继续 |
| 左右箭头 / `,` `.` | 前后逐帧，自动暂停 |
| `+` / `-` | 加速、减速 |
| R | 回到起始帧 |
| C | 显示物理碰撞几何 |
| L | 显示 link 名称 |
| M | 切换自碰撞颜色标记 |

碰撞报告分别列出机器人自碰撞、地面接触与环境接触。HUD 显示 link 对和穿透毫米数。
红色为至少 2 mm 的自穿透，橙色为更小穿透。被模型排除的 link 对会列出，不会假装已检查。

## 3. 批量处理、封存与复核

```bash
q1-frax-retarget --pattern 'walk*.bvh' --devices 4 --out outputs/walking
q1-frax-seal --dataset outputs/walking --held-out walk1_subject2 walk2_subject4
q1-frax-seal --dataset outputs/walking --verify-only
```

完整源动作分配到 train 或 eval，同一源动作不能出现在两侧。封存会核对 SHA256、
逐帧重放实际导出 CSV、重算源动作跟踪误差与碰撞证据，并检查每段至少连续 3 秒有效。
不满足要求会退出失败，不会把缺失指标填成 0。封存后不能覆盖；调参必须使用新目录。

质量文件中的 `reject_*` 保存逐帧拒收原因。原始 NPZ、未封存候选 CSV 和封存通过的
CSV 是不同状态。只有通过封存校验的片段才会显示 `SEALED / ACCEPTED REFERENCE`。
`--integrity-only` 只复核文件完整性，不代表重新通过运动质量验收。

## 4. 调参

`--config` 接收重定向 JSON，`--quality-profile` 接收完整版本化质量标准。
优先调足部位置/姿态权重、阻尼、摆脚间隙、自碰撞间隙和约束行数，保留完整参数记录。
不能只看动画或通过率；同时比较穿透、滑动、跟踪误差、速度/加速度和保留时长。
详细定义见 [质量标准](QUALITY.md) 和 [碰撞参数](COLLISIONS.md)。

桌面播放需要可用的 OpenGL 显示会话。无桌面渲染默认尝试 EGL；若主机仅支持 OSMesa，
安装相应驱动后可设置 `MUJOCO_GL=osmesa`。渲染程序不会为了画面平滑修改动作或隐藏失败帧。
