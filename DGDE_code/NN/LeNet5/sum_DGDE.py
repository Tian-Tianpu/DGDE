import os
import re
import joblib
import pandas as pd
import openpyxl


def summarize_DGDE_results(folder_path, output_excel):
    """
    遍历指定文件夹中的 DGDE .pkl 文件，提取参数及最后一个 epoch 的最优数据并保存至 Excel。

    参数规则: results_lr0.1_K01.3_f_p0.5_DGDE.pkl
    """
    results_list = []

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 {folder_path} 不存在。")
        return

    print(f"开始扫描目录: {folder_path} ...")

    # 使用 os.walk 递归查找所有子文件夹下的 pkl
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            # 匹配符合 DGDE 命名规则的文件
            if filename.endswith(".pkl") and "results_lr" in filename:
                file_path = os.path.join(root, filename)

                try:
                    # 1. 加载数据
                    data = joblib.load(file_path)

                    # 2. 使用正则表达式提取参数
                    # 匹配规则: lr([\d.]+)_K0([\d.]+)_f_p([\d.]+)
                    match = re.search(r"results_lr([\d.]+)_K0([\d.]+)_f_p([\d.]+)_DGDE\.pkl", filename)

                    if match:
                        lr = float(match.group(1))
                        k0 = float(match.group(2))
                        f_p = float(match.group(3))

                        # 3. 提取最后一个 Epoch 的全局最优数据
                        # 使用 .get(key, []) 防止键值不存在导致报错
                        b_v_losses = data.get("best_valid_losses", [])
                        b_v_accs = data.get("best_valid_accs", [])
                        b_t_losses = data.get("best_test_losses", [])
                        b_t_accs = data.get("best_test_accs", [])

                        # 整合单条记录
                        result_dict = {
                            "Algorithm": "DGDE",
                            "Dataset": os.path.basename(root),
                            "Learning_Rate": lr,
                            "K0": k0,
                            "fitness_p": f_p,
                            "Final_Valid_Loss": b_v_losses[-1] if b_v_losses else None,
                            "Final_Valid_Acc": b_v_accs[-1] if b_v_accs else None,
                            "Final_Test_Loss": b_t_losses[-1] if b_t_losses else None,
                            "Final_Test_Acc": b_t_accs[-1] if b_t_accs else None,
                            "Filename": filename
                        }
                        results_list.append(result_dict)
                    else:
                        print(f"跳过格式不符的文件: {filename}")

                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")

    # 4. 导出为 Excel
    if results_list:
        df = pd.DataFrame(results_list)
        # 按照参数排序，方便查看规律
        df = df.sort_values(by=["Learning_Rate", "K0", "fitness_p"])

        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='DGDE_Results', index=False)

        print("-" * 30)
        print(f"成功汇总 {len(results_list)} 条实验数据！")
        print(f"结果已保存至: {output_excel}")
    else:
        print("未发现有效的 .pkl 结果文件。")


if __name__ == "__main__":
    # 设定你的路径
    target_folder = "./Results_DGDE_SGD/MNIST_1000"
    output_file = "./Results_DGDE_SGD/DGDE_parameter_summary.xlsx"

    # 执行汇总
    summarize_DGDE_results(target_folder, output_file)