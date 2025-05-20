# Project 1: Parameter Estimation for Pyridine Problem

## 项目简介
- 基于多重射击法实现参数估计（Pyridine问题）
- 支持通用CNLLS优化（IPOPT & Gauß-Newton）

## 大纲
project1_pyridine/
├── src/
│   ├── dynamics.py                # 定义任意ODE系统（Pyridine模型）
│   ├── multiple_shooting.py       # 多重射击离散化
│   ├── cnlls_solver.py            # 通用CNLLS优化求解器（IPOPT & Gauß-Newton）
│   ├── sensitivity.py             # 灵敏度分析（CasADi实现）
│   └── utils.py                   # 数据生成与处理函数
├── data/
│   └── measurements.npy           # 测量数据
├── notebooks/
│   ├── milestone1_multiple_shooting.ipynb  # 里程碑1 (离散化)
│   ├── milestone2_cnlls_solver.ipynb       # 里程碑2 (优化求解器)
│   ├── milestone3_pyridine_problem.ipynb   # 里程碑3 (完整应用)
│   └── milestone4_presentation.ipynb       # 里程碑4 (结果展示)
├── results/                        # 存放运行结果、图表
├── environment.yml                 # conda环境定义
└── README.md                       # 项目说明文档
<img width="481" alt="Screenshot 2025-05-20 at 16 08 24" src="https://github.com/user-attachments/assets/26caf8d5-9ae3-4bc7-a655-eab4601f2a3b" />

## 项目实施顺序
1. **环境搭建与基础模块实现**  
   - 环境安装 (`environment.yml`)  
   - `dynamics.py` 和 `multiple_shooting.py` 基本实现

2. **优化问题求解器实现**  
   - `cnlls_solver.py`实现IPOPT和Gauß-Newton方法  
   - 灵敏度模块`sensitivity.py`集成

3. **完整的Pyridine问题测试**  
   - 数据生成(`utils.py`)  
   - milestone3笔记本进行测试

4. **整理结果与展示**  
   - milestone4 完成最终结果展示、分析与汇报准备
