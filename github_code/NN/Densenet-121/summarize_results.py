import joblib
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import openpyxl

def plot_curves(filename):
    """加载 .pkl 文件并绘制验证集、测试集损失、精度的变化曲线。

    Args:
        filename (str): .pkl 文件名。
    """
    try:
        results = joblib.load(filename)

        valid_losses = results["best_valid_losses"]
        valid_accs = results["best_valid_accs"]
        test_losses = results["best_test_losses"]
        test_accs = results["best_test_accs"]

        epochs = range(1, len(valid_losses) + 1)

        # 绘制损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, valid_losses, label="Validation Loss")
        plt.plot(epochs, test_losses, label="Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Validation and Test Loss Curves")
        plt.legend()
        plt.show()

        # 绘制精度曲线
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, valid_accs, label="Validation Accuracy")
        plt.plot(epochs, test_accs, label="Test Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Validation and Test Accuracy Curves")
        plt.legend()
        plt.show()

    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def plot_combined_curves(filename):
    """加载 .pkl 文件并绘制五个个体的验证集、测试集损失、精度的变化曲线（组合在一个图中）。

    Args:
        filename (str): .pkl 文件名。
    """
    try:
        results = joblib.load(filename)

        valid_losses = results["valid_losses"]
        valid_accs = results["valid_accs"]
        test_losses = results["test_losses"]
        test_accs = results["test_accs"]

        num_individuals = len(valid_losses[0])
        epochs = range(1, len(valid_losses) + 1)

        # 绘制验证集损失曲线
        plt.figure(figsize=(10, 5))
        for i in range(num_individuals):
            individual_valid_losses = [losses[i] for losses in valid_losses]
            plt.plot(epochs, individual_valid_losses, label=f"Individual {i+1}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Validation Loss Curves")
        plt.legend()
        plt.show()

        # 绘制测试集损失曲线
        plt.figure(figsize=(10, 5))
        for i in range(num_individuals):
            individual_test_losses = [losses[i] for losses in test_losses]
            plt.plot(epochs, individual_test_losses, label=f"Individual {i+1}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Test Loss Curves")
        plt.legend()
        plt.show()

        # 绘制验证集精度曲线
        plt.figure(figsize=(10, 5))
        for i in range(num_individuals):
            individual_valid_accs = [accs[i] for accs in valid_accs]
            plt.plot(epochs, individual_valid_accs, label=f"Individual {i+1}")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy Curves")
        plt.legend()
        plt.show()

        # 绘制测试集精度曲线
        plt.figure(figsize=(10, 5))
        for i in range(num_individuals):
            individual_test_accs = [accs[i] for accs in test_accs]
            plt.plot(epochs, individual_test_accs, label=f"Individual {i+1}")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Test Accuracy Curves")
        plt.legend()
        plt.show()

    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
#
# if __name__ == "__main__":
#     filename = "./Results_AdaptiveGDE/lenet5/MNIST_1000/results_0.001_0.3_0.7_0.1_1_0.1.pkl"
#     plot_curves(filename)
#
#     plot_combined_curves(filename)



def summarize_AdaptiveGDE_experiment_results(folder_path, excel_filename):
    """
    遍历指定文件夹中的所有 .pkl 文件，提取实验结果和参数，
    并将它们汇总到一个 Excel 文件中。

    Args:
        folder_path (str): 包含 .pkl 文件的文件夹路径。
        excel_filename (str, optional): 输出 Excel 文件的路径。
            默认为 "./Result_AdaptiveGDE/parameter_results.xlsx"。
    """

    results_list = []

    for dataname in os.listdir(folder_path):
        dataname_path = os.path.join(folder_path, dataname)
        if os.path.isdir(dataname_path):  # 确保是文件夹
            for filename in os.listdir(dataname_path):
                if filename.endswith(".pkl") and filename.startswith("results_"):
                    file_path = os.path.join(dataname_path, filename)
                    try:
                        results = joblib.load(file_path)

                        # 使用正则表达式从文件名中提取参数
                        match = re.match(r"results_([\d.]+)_([\d.]+)_([\d.]+)_([\d.]+)_([\d.]+)_([\d.]+)\.pkl", filename)
                        if match:
                            learning_rate, F_min, F_max, distance_factor_min, distance_factor_max, fitness_p = map(float, match.groups())

                            # 假设 .pkl 文件中包含以下结果列表
                            best_test_accs = results.get("best_test_accs", [])
                            best_test_losses = results.get("best_test_losses", [])
                            best_valid_accs = results.get("best_valid_accs", [])
                            best_valid_losses = results.get("best_valid_losses", [])
                            total_time = results.get("epoch_times")

                            # 提取最后一个元素（假设是最终结果）
                            best_test_acc = best_test_accs[-1] if best_test_accs else None
                            best_test_loss = best_test_losses[-1] if best_test_losses else None
                            best_valid_acc = best_valid_accs[-1] if best_valid_accs else None
                            best_valid_loss = best_valid_losses[-1] if best_valid_losses else None

                            result_dict = {
                                "algorithm": "AdaptiveGDE",
                                "datasets": dataname,
                                "learning_rate": learning_rate,
                                "F_min": F_min,
                                "F_max": F_max,
                                "distance_factor_min": distance_factor_min,
                                "distance_factor_max": distance_factor_max,
                                "fitness_p": fitness_p,
                                "best_test_accs": best_test_acc,
                                "best_test_losses": best_test_loss,
                                "best_valid_accs": best_valid_acc,
                                "best_valid_losses": best_valid_loss,
                                # "total_time": total_time
                            }
                            results_list.append(result_dict)
                        else:
                            print(f"Warning: Filename {filename} does not match expected pattern.")

                    except Exception as e:
                        print(f"Error processing {filename}: {e}")

    if results_list:
        df = pd.DataFrame(results_list)
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Experiment Results', index=False)
        print(f"Experiment results saved to {excel_filename}")
    else:
        print("No valid .pkl files found in the specified folder.")

def summarize_GDs_experiment_results(folder_path, excel_filename):
    results_list = []

    for root, dirs, files in os.walk(folder_path):  # 使用os.walk
        for filename in files:
            if filename.endswith(".pkl") and filename.startswith("results_"):
                file_path = os.path.join(root, filename)  # 使用root，而不是dataname_path
                try:
                    results = joblib.load(file_path)

                    # 使用正则表达式从文件名中提取算法名称和学习率
                    match = re.match(r"results_([\d.]+)_([A-Za-z]+)\.pkl", filename)
                    if match:
                        learning_rate, algorithm_name = float(match.group(1)), match.group(2)

                        # 提取实验结果
                        best_test_accs = results.get("best_test_accuracies", [])
                        best_test_losses = results.get("best_test_losses", [])
                        best_valid_accs = results.get("best_val_accuracies", [])
                        best_valid_losses = results.get("best_val_losses", [])
                        total_time = results.get("population_times")

                        best_test_acc = best_test_accs[-1] if best_test_accs else None
                        best_test_loss = best_test_losses[-1] if best_test_losses else None
                        best_valid_acc = best_valid_accs[-1] if best_valid_accs else None
                        best_valid_loss = best_valid_losses[-1] if best_valid_losses else None

                        result_dict = {
                            "algorithm": algorithm_name,
                            "datasets": os.path.basename(root),
                            "learning_rate": learning_rate,
                            "best_test_accs": best_test_acc,
                            "best_test_losses": best_test_loss,
                            "best_valid_accs": best_valid_acc,
                            "best_valid_losses": best_valid_loss,
                            # "total_time": total_time
                        }
                        results_list.append(result_dict)
                    else:
                        print(f"Warning: Filename {filename} does not match expected pattern.")

                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    if results_list:
        df = pd.DataFrame(results_list)
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Experiment Results', index=False)
        print(f"Experiment results saved to {excel_filename}")
    else:
        print("No valid .pkl files found in the specified folder.")


def summarize_ESGD_experiment_results(folder_path, excel_filename):
    all_data = []

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".pkl") and ("ESGD_ADAM" in filename or "ESGD_SGD" in filename):
                file_path = os.path.join(root, filename)

                try:
                    results = joblib.load(file_path)

                    # 使用正则表达式从文件名中提取关键信息
                    match = re.match(r"results_([\d.]+)_([A-Za-z0-9.]+)_(ESGD_ADAM|ESGD_SGD)\.pkl", filename)
                    if match:
                        learning_rate, F, algorithm_name = float(match.group(1)), match.group(2), match.group(3)

                        # 提取实验结果
                        best_test_accs = results.get("best_test_accs", [])
                        best_test_losses = results.get("best_test_losses", [])
                        best_valid_accs = results.get("best_valid_accs", [])
                        best_valid_losses = results.get("best_valid_losses", [])
                        total_time = results.get("epoch_times")

                        # 提取最后一个元素（假设是最终结果）
                        best_test_acc = best_test_accs[-1] if best_test_accs else None
                        best_test_loss = best_test_losses[-1] if best_test_losses else None
                        best_valid_acc = best_valid_accs[-1] if best_valid_accs else None
                        best_valid_loss = best_valid_losses[-1] if best_valid_losses else None

                        # 组织数据
                        data_entry = {
                            # "filename": filename,
                            "algorithm": algorithm_name,
                            "datasets": os.path.basename(root),
                            "learning_rate": learning_rate,
                            "F": F,
                            "best_test_accs": best_test_acc,
                            "best_test_losses": best_test_loss,
                            "best_valid_accs": best_valid_acc,
                            "best_valid_losses": best_valid_loss,
                            # "train_losses": results.get("train_losses"),
                            # "test_losses": results.get("test_losses"),
                            # "valid_losses": results.get("valid_losses"),
                            # "train_accs": results.get("train_accs"),
                            # "test_accs": results.get("test_accs"),
                            # "valid_accs": results.get("valid_accs"),
                            # "epoch_times": results.get("epoch_times"),
                            # "total_time": total_time
                            # "weight_individuals": results.get("weight_individuals"),
                            # "best_train_accs": results.get("best_train_accs"),
                            # "best_test_accs": results.get("best_test_accs"),
                            # "best_valid_accs": results.get("best_valid_accs"),
                            # "num_niches": results.get("num_niches")
                        }

                        # # 处理字典中的稀疏权重
                        # for threshold, values in results.get("sparsity_weights", {}).items():
                        #     data_entry[f"sparsity_weights_{threshold}"] = values
                        #
                        # # 处理字典中的稀疏性变化
                        # for threshold, values in results.get("sparsity_epochs", {}).items():
                        #     data_entry[f"sparsity_epochs_{threshold}"] = values

                        all_data.append(data_entry)
                    else:
                        print(f"Warning: Filename {filename} does not match expected pattern.")

                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    # 转换为 DataFrame 并存储
    if all_data:
        df = pd.DataFrame(all_data)
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Experiment Results', index=False)
        print(f"Experiment results saved to {excel_filename}")
    else:
        print("No valid .pkl files found in the specified folder.")

def summarize_NGDE_experiment_results(folder_path, excel_filename):
    all_data = []

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".pkl") and ("NGDE_ADAM" in filename or "NGDE_SGD" in filename):
                file_path = os.path.join(root, filename)

                try:
                    results = joblib.load(file_path)

                    # 使用正则表达式从文件名中提取关键信息
                    match = re.match(r"results_([\d.]+)_([A-Za-z0-9.]+)_([\d.]+)_(NGDE_ADAM|NGDE_SGD)\.pkl", filename)
                    if match:
                        learning_rate, F, fitness_p, algorithm_name = float(match.group(1)), match.group(2), float(match.group(3)), match.group(4)

                        # 提取实验结果
                        best_test_accs = results.get("best_test_accs", [])
                        best_test_losses = results.get("best_test_losses", [])
                        best_valid_accs = results.get("best_valid_accs", [])
                        best_valid_losses = results.get("best_valid_losses", [])
                        total_time = results.get("epoch_times")

                        # 提取最后一个元素（假设是最终结果）
                        best_test_acc = best_test_accs[-1] if best_test_accs else None
                        best_test_loss = best_test_losses[-1] if best_test_losses else None
                        best_valid_acc = best_valid_accs[-1] if best_valid_accs else None
                        best_valid_loss = best_valid_losses[-1] if best_valid_losses else None

                        data_entry = {
                            "algorithm": algorithm_name,
                            "datasets": os.path.basename(root),
                            "learning_rate": learning_rate,
                            "F": F,
                            "fitness_p": fitness_p,
                            "best_test_accs": best_test_acc,
                            "best_test_losses": best_test_loss,
                            "best_valid_accs": best_valid_acc,
                            "best_valid_losses": best_valid_loss,
                            # "epoch_times": results.get("epoch_times"),
                        }

                        # # 处理字典中的稀疏权重
                        # for threshold, values in results.get("sparsity_weights", {}).items():
                        #     data_entry[f"sparsity_weights_{threshold}"] = values
                        #
                        # # 处理字典中的稀疏性变化
                        # for threshold, values in results.get("sparsity_epochs", {}).items():
                        #     data_entry[f"sparsity_epochs_{threshold}"] = values

                        all_data.append(data_entry)
                    else:
                        print(f"Warning: Filename {filename} does not match expected pattern.")

                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    # 转换为 DataFrame 并存储
    if all_data:
        df = pd.DataFrame(all_data)
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Experiment Results', index=False)
        print(f"Experiment results saved to {excel_filename}")
    else:
        print("No valid .pkl files found in the specified folder.")

# def summarize_ESGD_experiment_results(folder_path, excel_filename):
#     all_data = []
#
#     for root, dirs, files in os.walk(folder_path):
#         for filename in files:
#             if filename.endswith(".pkl") and ("ESGD_ADAM" in filename or "ESGD_SGD" in filename):
#                 file_path = os.path.join(root, filename)
#
#                 try:
#                     results = joblib.load(file_path)
#
#                     # 使用正则表达式从文件名中提取关键信息
#                     match = re.match(r"results_([\d.]+)_([A-Za-z0-9.]+)_ESGD_(ADAM|SGD)\.pkl", filename)
#                     if match:
#                         learning_rate, F, optimizer = float(match.group(1)), match.group(2), match.group(3)
#
#                         # 组织数据
#                         data_entry = {
#                             "filename": filename,
#                             "learning_rate": learning_rate,
#                             "F": F,
#                             "optimizer": optimizer,
#                             "train_losses": results.get("train_losses"),
#                             "test_losses": results.get("test_losses"),
#                             "valid_losses": results.get("valid_losses"),
#                             "train_accs": results.get("train_accs"),
#                             "test_accs": results.get("test_accs"),
#                             "valid_accs": results.get("valid_accs"),
#                             "epoch_times": results.get("epoch_times"),
#                             "weight_individuals": results.get("weight_individuals"),
#                             "best_train_accs": results.get("best_train_accs"),
#                             "best_test_accs": results.get("best_test_accs"),
#                             "best_valid_accs": results.get("best_valid_accs"),
#                             "num_niches": results.get("num_niches")
#                         }
#
#                         # 处理字典中的稀疏权重
#                         for threshold, values in results.get("sparsity_weights", {}).items():
#                             data_entry[f"sparsity_weights_{threshold}"] = values
#
#                         # 处理字典中的稀疏性变化
#                         for threshold, values in results.get("sparsity_epochs", {}).items():
#                             data_entry[f"sparsity_epochs_{threshold}"] = values
#
#                         all_data.append(data_entry)
#                     else:
#                         print(f"Warning: Filename {filename} does not match expected pattern.")
#
#                 except Exception as e:
#                     print(f"Error processing {filename}: {e}")
#
#     # 转换为 DataFrame 并存储
#     if all_data:
#         df = pd.DataFrame(all_data)
#         with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
#             df.to_excel(writer, sheet_name='Experiment Results', index=False)
#         print(f"Experiment results saved to {excel_filename}")
#     else:
#         print("No valid .pkl files found in the specified folder.")
#
# def summarize_NGDE_experiment_results(folder_path, excel_filename):
#     all_data = []
#
#     for root, dirs, files in os.walk(folder_path):
#         for filename in files:
#             if filename.endswith(".pkl") and ("NGDE_ADAM" in filename or "NGDE_SGD" in filename):
#                 file_path = os.path.join(root, filename)
#
#                 try:
#                     results = joblib.load(file_path)
#
#                     # 使用正则表达式从文件名中提取关键信息
#                     match = re.match(r"results_([\d.]+)_([A-Za-z0-9.]+)_([\d.]+)_NGDE_(ADAM|SGD)\.pkl", filename)
#                     if match:
#                         learning_rate, F, fitness_p, optimizer = float(match.group(1)), match.group(2), float(match.group(3)), match.group(4)
#
#                         # 组织数据
#                         data_entry = {
#                             "filename": filename,
#                             "learning_rate": learning_rate,
#                             "F": F,
#                             "fitness_p": fitness_p,
#                             "optimizer": optimizer,
#                             "train_losses": results.get("train_losses"),
#                             "test_losses": results.get("test_losses"),
#                             "valid_losses": results.get("valid_losses"),
#                             "train_accs": results.get("train_accs"),
#                             "test_accs": results.get("test_accs"),
#                             "valid_accs": results.get("valid_accs"),
#                             "epoch_times": results.get("epoch_times"),
#                             "weight_individuals": results.get("weight_individuals"),
#                             "best_train_accs": results.get("best_train_accs"),
#                             "best_test_accs": results.get("best_test_accs"),
#                             "best_valid_accs": results.get("best_valid_accs"),
#                             "num_niches": results.get("num_niches")
#                         }
#
#                         # 处理字典中的稀疏权重
#                         for threshold, values in results.get("sparsity_weights", {}).items():
#                             data_entry[f"sparsity_weights_{threshold}"] = values
#
#                         # 处理字典中的稀疏性变化
#                         for threshold, values in results.get("sparsity_epochs", {}).items():
#                             data_entry[f"sparsity_epochs_{threshold}"] = values
#
#                         all_data.append(data_entry)
#                     else:
#                         print(f"Warning: Filename {filename} does not match expected pattern.")
#
#                 except Exception as e:
#                     print(f"Error processing {filename}: {e}")
#
#     # 转换为 DataFrame 并存储
#     if all_data:
#         df = pd.DataFrame(all_data)
#         with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
#             df.to_excel(writer, sheet_name='Experiment Results', index=False)
#         print(f"Experiment results saved to {excel_filename}")
#     else:
#         print("No valid .pkl files found in the specified folder.")


folder_path_adaptiveGDE = "./Results_AdaptiveGDE/"
excel_adaptiveGDE_filepath = "./Results_AdaptiveGDE/parameter_results.xlsx"
summarize_AdaptiveGDE_experiment_results(folder_path_adaptiveGDE, excel_adaptiveGDE_filepath)

folder_path_ngde = "./Results_NGDE/"
excel_ngde_filepath = "./Results_NGDE/parameter_results.xlsx"
summarize_NGDE_experiment_results(folder_path_ngde, excel_ngde_filepath)

folder_path_gd = "./Results_GD/"
excel_gd_filename = "./Results_GD/parameter_results.xlsx"
summarize_GDs_experiment_results(folder_path_gd, excel_gd_filename)


folder_path_esgd = "./Results_ESGD/"
excel_esgd_filepath = "./Results_ESGD/parameter_results.xlsx"
summarize_ESGD_experiment_results(folder_path_esgd, excel_esgd_filepath)




def combine_all_experiment_results(output_excel, result_folders):
    """
    将多个文件夹中的所有 Excel 文件合并到一个 Excel 文件中，
    即使某些文件缺少一些列，也会保留所有的列。

    Args:
        output_excel (str): 输出的 Excel 文件路径。
        result_folders (list): 含有实验结果文件夹路径的列表。
    """
    all_data = []

    # 遍历每个文件夹
    for folder_path in result_folders:
        for filename in os.listdir(folder_path):
            if filename.endswith(".xlsx") and "parameter_results" in filename:
                file_path = os.path.join(folder_path, filename)
                df = pd.read_excel(file_path, sheet_name="Experiment Results")

                all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True, sort=False)

    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        combined_df.to_excel(writer, sheet_name="All_Experiment_Results", index=False)

    print(f"All experiment results have been merged into {output_excel}")


result_folders = ["./Results_GD", "./Results_AdaptiveGDE", "./Results_ESGD", "./Results_NGDE"]
combine_all_experiment_results("all_experiment_results.xlsx", result_folders)


# def summarize_AdaptiveGDE_experiment_results(folder_path, excel_filename):
#     """
#     遍历指定文件夹中的所有 .pkl 文件，提取实验结果和参数，
#     并将它们汇总到一个 Excel 文件中。
#
#     Args:
#         folder_path (str): 包含 .pkl 文件的文件夹路径。
#         excel_filename (str, optional): 输出 Excel 文件的路径。
#             默认为 "./Result_AdaptiveGDE/parameter_results.xlsx"。
#     """
#
#     results_list = []
#
#     for dataname in os.listdir(folder_path):
#         dataname_path = os.path.join(folder_path, dataname)
#         if os.path.isdir(dataname_path):  # 确保是文件夹
#             for filename in os.listdir(dataname_path):
#                 if filename.endswith(".pkl") and filename.startswith("results_"):
#                     file_path = os.path.join(dataname_path, filename)
#                     try:
#                         results = joblib.load(file_path)
#
#                         # 使用正则表达式从文件名中提取参数
#                         match = re.match(r"results_([\d.]+)_([\d.]+)_([\d.]+)_([\d.]+)_([\d.]+)_([\d.]+)\.pkl", filename)
#                         if match:
#                             learning_rate, F_min, F_max, distance_factor_min, distance_factor_max, fitness_p = map(float, match.groups())
#
#                             # 假设 .pkl 文件中包含以下结果列表
#                             best_test_accs = results.get("best_test_accs", [])
#                             best_test_losses = results.get("best_test_losses", [])
#                             best_valid_accs = results.get("best_valid_accs", [])
#                             best_valid_losses = results.get("best_valid_losses", [])
#                             total_time = results.get("epoch_times")
#
#                             # 提取最后一个元素（假设是最终结果）
#                             best_test_acc = best_test_accs[-1] if best_test_accs else None
#                             best_test_loss = best_test_losses[-1] if best_test_losses else None
#                             best_valid_acc = best_valid_accs[-1] if best_valid_accs else None
#                             best_valid_loss = best_valid_losses[-1] if best_valid_losses else None
#
#                             result_dict = {
#                                 "algorithm": "AdaptiveGDE",
#                                 "datasets": dataname,
#                                 "learning_rate": learning_rate,
#                                 "F_min": F_min,
#                                 "F_max": F_max,
#                                 "distance_factor_min": distance_factor_min,
#                                 "distance_factor_max": distance_factor_max,
#                                 "fitness_p": fitness_p,
#                                 "best_test_accs": best_test_acc,
#                                 "best_test_losses": best_test_loss,
#                                 "best_valid_accs": best_valid_acc,
#                                 "best_valid_losses": best_valid_loss,
#                                 "total_time": total_time
#                             }
#                             results_list.append(result_dict)
#                         else:
#                             print(f"Warning: Filename {filename} does not match expected pattern.")
#
#                     except Exception as e:
#                         print(f"Error processing {filename}: {e}")
#
#     if results_list:
#         df = pd.DataFrame(results_list)
#         with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
#             df.to_excel(writer, sheet_name='Experiment Results', index=False)
#         print(f"Experiment results saved to {excel_filename}")
#     else:
#         print("No valid .pkl files found in the specified folder.")
#
# def summarize_GDs_experiment_results(folder_path, excel_filename):
#     results_list = []
#
#     for root, dirs, files in os.walk(folder_path): # 使用os.walk
#         for filename in files:
#             if filename.endswith(".pkl") and filename.startswith("results_"):
#                 file_path = os.path.join(root, filename) # 使用root，而不是dataname_path
#                 try:
#                     results = joblib.load(file_path)
#
#                     # 使用正则表达式从文件名中提取算法名称和学习率
#                     match = re.match(r"results_([\d.]+)_([A-Za-z]+)\.pkl", filename)
#                     if match:
#                         learning_rate, algorithm_name = float(match.group(1)), match.group(2)
#
#                         # 提取实验结果
#                         best_test_accs = results.get("best_test_accuracies", [])
#                         best_test_losses = results.get("best_test_losses", [])
#                         best_valid_accs = results.get("best_val_accuracies", [])
#                         best_valid_losses = results.get("best_val_losses", [])
#                         total_time = results.get("total_time")
#
#                         # 提取最后一个元素（假设是最终结果）
#                         best_test_acc = best_test_accs[-1] if best_test_accs else None
#                         best_test_loss = best_test_losses[-1] if best_test_losses else None
#                         best_valid_acc = best_valid_accs[-1] if best_valid_accs else None
#                         best_valid_loss = best_valid_losses[-1] if best_valid_losses else None
#
#                         result_dict = {
#                             "algorithm": algorithm_name,
#                             "datasets": os.path.basename(root),
#                             "learning_rate": learning_rate,
#                             "best_test_accs": best_test_acc,
#                             "best_test_losses": best_test_loss,
#                             "best_valid_accs": best_valid_acc,
#                             "best_valid_losses": best_valid_loss,
#                             "total_time": total_time
#                         }
#                         results_list.append(result_dict)
#                     else:
#                         print(f"Warning: Filename {filename} does not match expected pattern.")
#
#                 except Exception as e:
#                     print(f"Error processing {filename}: {e}")
#
#     if results_list:
#         df = pd.DataFrame(results_list)
#         with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
#             df.to_excel(writer, sheet_name='Experiment Results', index=False)
#         print(f"Experiment results saved to {excel_filename}")
#     else:
#         print("No valid .pkl files found in the specified folder.")
#
#
# def summarize_ESGD_experiment_results(folder_path, output_csv):
#     all_data = []
#
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".pkl") and ("ESGD_ADAM" in filename or "ESGD_SGD" in filename):
#             file_path = os.path.join(folder_path, filename)
#
#             with open(file_path, "rb") as f:
#                 results = joblib.load(f)
#
#                 # 提取文件名中的关键信息
#                 parts = filename.split("_")
#                 learning_rate = parts[1]
#                 F = parts[2]
#                 optimizer = "ADAM" if "ESGD_ADAM" in filename else "SGD"
#
#                 # 组织数据
#                 data_entry = {
#                     "filename": filename,
#                     "learning_rate": learning_rate,
#                     "F": F,
#                     "optimizer": optimizer,
#                     "train_losses": results["train_losses"],
#                     "test_losses": results["test_losses"],
#                     "valid_losses": results["valid_losses"],
#                     "train_accs": results["train_accs"],
#                     "test_accs": results["test_accs"],
#                     "valid_accs": results["valid_accs"],
#                     "epoch_times": results["epoch_times"],
#                     "weight_individuals": results["weight_individuals"],
#                     "best_train_accs": results["best_train_accs"],
#                     "best_test_accs": results["best_test_accs"],
#                     "best_valid_accs": results["best_valid_accs"],
#                     "num_niches": results["num_niches"]
#                 }
#
#                 # 处理字典中的稀疏权重
#                 for threshold, values in results["sparsity_weights"].items():
#                     data_entry[f"sparsity_weights_{threshold}"] = values
#
#                 # 处理字典中的稀疏性变化
#                 for threshold, values in results["sparsity_epochs"].items():
#                     data_entry[f"sparsity_epochs_{threshold}"] = values
#
#                 all_data.append(data_entry)
#
#     # 转换为 DataFrame 并存储
#     df = pd.DataFrame(all_data)
#     df.to_csv(output_csv, index=False)
#     print(f"CSV 文件已保存至: {output_csv}")
#
#
# def summarize_NGDE_experiment_results(folder_path, output_csv):
#     all_data = []
#
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".pkl") and ("NGDE_ADAM" in filename or "NGDE_SGD" in filename):
#             file_path = os.path.join(folder_path, filename)
#
#             with open(file_path, "rb") as f:
#                 results = joblib.load(f)
#
#                 # 提取文件名中的关键信息
#                 parts = filename.split("_")
#                 learning_rate = parts[1]
#                 F = parts[2]
#                 fitness_p = parts[3]
#                 optimizer = "ADAM" if "ADAM" in filename else "SGD"
#
#                 # 组织数据
#                 data_entry = {
#                     "filename": filename,
#                     "learning_rate": learning_rate,
#                     "F": F,
#                     "fitness_p": fitness_p,
#                     "optimizer": optimizer,
#                     "train_losses": results["train_losses"],
#                     "test_losses": results["test_losses"],
#                     "valid_losses": results["valid_losses"],
#                     "train_accs": results["train_accs"],
#                     "test_accs": results["test_accs"],
#                     "valid_accs": results["valid_accs"],
#                     "epoch_times": results["epoch_times"],
#                     "weight_individuals": results["weight_individuals"],
#                     "best_train_accs": results["best_train_accs"],
#                     "best_test_accs": results["best_test_accs"],
#                     "best_valid_accs": results["best_valid_accs"],
#                     "num_niches": results["num_niches"]
#                 }
#
#                 # 处理字典中的稀疏权重
#                 for threshold, values in results["sparsity_weights"].items():
#                     data_entry[f"sparsity_weights_{threshold}"] = values
#
#                 # 处理字典中的稀疏性变化
#                 for threshold, values in results["sparsity_epochs"].items():
#                     data_entry[f"sparsity_epochs_{threshold}"] = values
#
#                 all_data.append(data_entry)
#
#     # 转换为 DataFrame 并存储
#     df = pd.DataFrame(all_data)
#     df.to_csv(output_csv, index=False)
#     print(f"CSV 文件已保存至: {output_csv}")
#
#
#
#
#
# folder_path = "./Results_GD/"
# excel_gd_filename = "./Results_GD/parameter_results.xlsx"
# summarize_GDs_experiment_results(folder_path, excel_gd_filename)
# #
# #
# folder_path = "./Results_AdaptiveGDE/"
# excel_adaptiveGDE_filepath = "./Results_AdaptiveGDE/parameter_results.xlsx"
# summarize_AdaptiveGDE_experiment_results(folder_path, excel_adaptiveGDE_filepath)
#
#
# folder_path = "./Results_ESGD/"
# excel_esgd_filepath = "./Results_ESGD/parameter_results.xlsx"
# summarize_ESGD_experiment_results(folder_path, excel_esgd_filepath)
#
#
# folder_path = "./Results_NGDE/"
# excel_esgd_filepath = "./Results_NGDE/parameter_results.xlsx"
# summarize_ESGD_experiment_results(folder_path, excel_esgd_filepath)



# output_file = os.path.abspath("./merged_results.csv")
# merge_excel_files(excel_adaptiveGDE_filepath, excel_gd_filename, output_file)