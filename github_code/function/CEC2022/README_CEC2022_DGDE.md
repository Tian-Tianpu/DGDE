# CEC2022 DGDE 调参工程

同目录文件：

```text
run_DGDE_CEC2022_function.py  调参入口
new_DGDE.py                  与CEC2014相同的两阶段投影DGDE
all_function_new.py          CEC2022函数选择和实例读写
CEC2022_0923.py              目标函数及梯度实现
estimated_gradient.py       F4数值差分梯度
test_cec2022_runner.py       回归测试
```

入口直接导入 `new_DGDE.DGDE` 和 `all_function_new.select_function`。
默认仅运行CEC2022的F1-F12；库中13-28为附加基础函数，不加入此次搜索。

## 运行

```bat
python -m pip install numpy scipy
python run_DGDE_CEC2022_function.py --dry-run
python run_DGDE_CEC2022_function.py --generations 3 --limit 12 --output smoke_cec2022
python run_DGDE_CEC2022_function.py --workers 4
```

参数网格与CEC2014完全相同：

- NP: 10, 20, 30, 40
- s: 3, 4, 5, 7
- pf: 0.01, 0.05, 0.1, 0.3, 0.5, 0.7
- F: 0.3, 0.5, 0.7
- K_base: 0.3, 0.5, 0.7
- K0: 1.1, 1.3, 1.5, 1.7
- eta: 0.01, 0.001, 0.0001, 0.00001, 0.000001

默认100维、100代、seed=0，每组一次；每函数17280组，总计207360次优化。
rho_high=rho_low=0.8，默认boundary=clip。
顶部Dimension、budget、seed、num_processes、FUNS、GRID可直接编辑。
支持 `--functions 1 2 3` 和 `--dimensions 100 1000 10000`。

## 数据、结果和续跑

实例默认在脚本旁的pkl_data_sparsity，可用 `--data-dir` 修改。
主进程先串行准备实例，再启动workers；已有文件保留，缺失文件按
`--instance-seed 42` 和函数/维度确定性生成。
F6-F8的shuffle文件与shift/rotation文件一起固定并记录SHA256。
缓存按数据目录区分，检查矩阵形状及shuffle排列的有效性。
沿用仓库中的块稀疏高维改编定义，不代表官方原始CEC2022实例。

结果默认保存到results_DGDE_CEC2022_tuning_projected/F01/100D至F12/100D。
文件命名、pickle字段、summary.csv、失败记录及断点续跑方式与CEC2014相同。
修改代码/数据/配置后使用新输出目录；`--retry-failed` 可重试失败任务。

NFE记录直接目标函数调用，NGE记录梯度回调调用。
F4采用中心差分（100/1000维）或随机坐标差分（10000维），会额外评估目标函数：
额外实际调用次数在gradient_internal_nfe记录，总数为total_objective_calls。
因此F4的总目标调用预算不同于纯解析梯度函数，不能只用NFE代表总成本。

本入口使用NumPy数组回调，原函数库中autograd.numpy基础运算改用等效NumPy运算，
不引入自动微分依赖；公式和现有梯度方法不作重新推导。
本次未验证全部CEC梯度的数学正确性，运行成功也不是准确性证明。

## 验证

100维、3代、双进程，覆盖全部12个函数，12/12成功。
回归测试检查网格数量、编号范围、实例/排列固定、矩阵形状、F4额外调用计数及续跑。
完整207360任务参数搜索尚未执行。
