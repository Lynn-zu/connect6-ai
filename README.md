# 六子棋智能博弈系统

这是一个基于 Flask 的六子棋 Web 对弈项目，支持玩家与机器人进行 15 × 15 或 19 × 19 棋盘对局。项目内置传统搜索机器人，也可以使用训练模型进行落子判断。

## 功能特点

- Web 棋盘对弈界面，支持桌面端和手机端访问
- 六子棋规则：黑方首回合一子，之后每回合两子
- 支持 15 × 15 标准棋盘和 19 × 19 大棋盘
- 支持悔棋、前进、重新开局、更换对手
- 支持训练机器人、增强机器人、搜索机器人三种对手
- 提供模型训练、训练状态查询和模型信息接口

## 技术栈

- Python 3.10 / 3.11
- Flask
- NumPy
- TensorFlow / Keras
- HTML / CSS / JavaScript

## 项目结构

```text
.
├── app.py                  # Web 服务、游戏接口和机器人逻辑
├── train.py                # 模型训练脚本
├── requirements.txt        # Python 依赖
├── templates/
│   └── index.html          # 前端页面
├── tests/
│   └── test_ai_regression.py
├── models/
│   ├── connect6.model.h5   # 当前默认训练模型，可选
│   └── training_meta.json  # 训练元信息，可选
└── data/                   # 训练数据与对局缓存，本地生成
```

## 环境准备

建议使用虚拟环境运行项目：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果本机没有 Python 3.11，也可以使用 Python 3.10。

## 启动项目

```powershell
.\.venv\Scripts\activate
python app.py
```

启动后打开浏览器访问：

```text
http://localhost:5000
```

## 模型说明

默认模型路径为：

```text
models/connect6.model.h5
```

如果该文件存在，15 × 15 棋盘可以使用训练机器人或增强机器人。  
如果模型不存在，项目仍然可以运行，系统会回退到传统搜索或启发式判断。

19 × 19 棋盘暂时只支持搜索机器人。

## 训练接口

项目提供以下训练相关接口：

| 接口 | 方法 | 说明 |
| --- | --- | --- |
| `/api/train` | POST | 启动后台训练 |
| `/api/train_model` | POST | 启动后台训练，兼容旧接口 |
| `/api/train_status` | GET | 查询训练状态 |
| `/api/model_info` | GET | 查询模型结构和模型文件状态 |
| `/api/download_model` | GET | 下载当前模型 |

训练接口可传入参数，例如：

```json
{
  "num_games": 20,
  "epochs": 3,
  "iterations": 1
}
```

## 运行测试

```powershell
pytest
```

## 上传 GitHub 前的建议

本项目会生成训练数据、缓存、历史模型和软著材料包，这些文件不建议直接提交到 GitHub。当前 `.gitignore` 已默认排除：

- 虚拟环境和 Python 缓存
- 测试缓存和本地日志
- `data/*.npz`、`data/games/`
- 软著生成材料和 zip 包
- 历史模型、备份模型

默认只允许提交：

```text
models/connect6.model.h5
models/training_meta.json
```

如果模型文件后续变得很大，建议使用 Git LFS 或 GitHub Release 管理模型文件。

## 常用 GitHub 上传命令

```powershell
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/你的用户名/你的仓库名.git
git push -u origin main
```

如果需要使用 Git LFS 管理模型文件：

```powershell
git lfs install
git lfs track "*.h5"
git add .gitattributes
git add models/connect6.model.h5
git commit -m "Add trained model"
git push
```
