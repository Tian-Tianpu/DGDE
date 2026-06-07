import joblib  # 使用 joblib
import matplotlib.pyplot as plt
import pandas as pd

def load_results(figure_save_path, filename):
    with open(figure_save_path + filename, "rb") as f:
        results = joblib.load(f)
    return results

def save_to_excel(figure_save_path):
    algorithms = ["AGDE", "MSGD", "MAdam", "MRMSprop", "MAdamW", "MRAdam", "MNAdam"]

    # 存储每一代的结果
    epoch_results_list = []
    for algorithm in algorithms:
        filename = f"{algorithm}.pkl"
        results = load_results(figure_save_path, filename)

        test_accuracies = results["best_test_accuracies"]
        test_losses = results["best_test_losses"]
        train_accuracies = results["population_train_accuracies"][0]
        train_losses = results["population_train_losses"][0]

        for epoch in range(len(test_accuracies)):
            epoch_results_list.append({
                "Algorithm": algorithm,
                "Epoch": epoch,
                "Test Accuracy": test_accuracies[epoch],
                "Test Loss": test_losses[epoch],
                "Train Accuracy": train_accuracies[epoch],
                "Train Loss": train_losses[epoch]
            })

    epoch_df = pd.DataFrame(epoch_results_list)
    epoch_df.to_excel(figure_save_path + "epoch_results.xlsx", index=False)

    # 存储最终结果和总运行时间
    final_results_list = []
    for algorithm in algorithms:
        filename = f"{algorithm}.pkl"
        results = load_results(figure_save_path, filename)

        test_accuracies = results["best_test_accuracies"]
        test_losses = results["best_test_losses"]
        total_time = results["total_time"]

        best_test_accuracy = max(test_accuracies)
        best_test_loss = min(test_losses)

        final_results_list.append({
            "Algorithm": algorithm,
            "Test Accuracy": best_test_accuracy,
            "Test Loss": best_test_loss,
            "Total Time": total_time
        })

    final_df = pd.DataFrame(final_results_list)
    final_df.to_excel(figure_save_path + "final_results.xlsx", index=False)

def plot_results(agde_results, msgd_results, madam_results, mrmsprop_results,
                 madamw_results, mradam_results, mnadam_results, figure_save_path):
    agde_test_accuracies = agde_results["best_test_accuracies"]
    agde_test_losses = agde_results["best_test_losses"]
    agde_train_accuracies = agde_results["population_train_accuracies"][0]
    agde_train_losses = agde_results["population_train_losses"][0]

    msgd_test_accuracies = msgd_results["best_test_accuracies"]
    msgd_test_losses = msgd_results["best_test_losses"]
    msgd_train_accuracies = msgd_results["population_train_accuracies"][0]
    msgd_train_losses = msgd_results["population_train_losses"][0]

    madam_test_accuracies = madam_results["best_test_accuracies"]
    madam_test_losses = madam_results["best_test_losses"]
    madam_train_accuracies = madam_results["population_train_accuracies"][0]
    madam_train_losses = madam_results["population_train_losses"][0]

    mrmsprop_test_accuracies = mrmsprop_results["best_test_accuracies"]
    mrmsprop_test_losses = mrmsprop_results["best_test_losses"]
    mrmsprop_train_accuracies = mrmsprop_results["population_train_accuracies"][0]
    mrmsprop_train_losses = mrmsprop_results["population_train_losses"][0]

    madamw_test_accuracies = madamw_results["best_test_accuracies"]
    madamw_test_losses = madamw_results["best_test_losses"]
    madamw_train_accuracies = madamw_results["population_train_accuracies"][0]
    madamw_train_losses = madamw_results["population_train_losses"][0]

    mradam_test_accuracies = mradam_results["best_test_accuracies"]
    mradam_test_losses = mradam_results["best_test_losses"]
    mradam_train_accuracies = mradam_results["population_train_accuracies"][0]
    mradam_train_losses = mradam_results["population_train_losses"][0]

    mnadam_test_accuracies = mnadam_results["best_test_accuracies"]
    mnadam_test_losses = mnadam_results["best_test_losses"]
    mnadam_train_accuracies = mnadam_results["population_train_accuracies"][0]
    mnadam_train_losses = mnadam_results["population_train_losses"][0]

    plt.figure(figsize=(9, 6))
    plt.plot(agde_test_accuracies, label="AGDE")
    plt.plot(msgd_test_accuracies, label="MSGD")
    plt.plot(madam_test_accuracies, label="MAdam")
    plt.plot(mrmsprop_test_accuracies, label="MRMSprop")
    plt.plot(madamw_test_accuracies, label="MAdamW")
    plt.plot(mradam_test_accuracies, label="MRAdam")
    plt.plot(mnadam_test_accuracies, label="MNAdam")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_save_path + "test_acc.png", dpi=300)
    # plt.show()

    # 绘制测试损失变化
    plt.figure(figsize=(9, 6))
    plt.plot(agde_test_losses, label="AGDE")
    plt.plot(msgd_test_losses, label="MSGD")
    plt.plot(madam_test_losses, label="MAdam")
    plt.plot(mrmsprop_test_losses, label="MRMSprop")
    plt.plot(madamw_test_losses, label="MAdamW")
    plt.plot(mradam_test_losses, label="MRAdam")
    plt.plot(mnadam_test_losses, label="MNAdam")
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_save_path + "test_loss.png", dpi=300)
    # plt.show()

    # 绘制训练精度变化
    plt.figure(figsize=(9, 6))
    plt.plot(agde_train_accuracies, label="AGDE")
    plt.plot(msgd_train_accuracies, label="MSGD")
    plt.plot(madam_train_accuracies, label="MAdam")
    plt.plot(mrmsprop_train_accuracies, label="MRMSprop")
    plt.plot(madamw_train_accuracies, label="MAdamW")
    plt.plot(mradam_train_accuracies, label="MRAdam")
    plt.plot(mnadam_train_accuracies, label="MNAdam")
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_save_path + "train_acc.png", dpi=300)
    # plt.show()

    # 绘制训练损失变化
    plt.figure(figsize=(9, 6))
    plt.plot(agde_train_losses, label="AGDE")
    plt.plot(msgd_train_losses, label="MSGD")
    plt.plot(madam_train_losses, label="MAdam")
    plt.plot(mrmsprop_train_losses, label="MRMSprop")
    plt.plot(madamw_train_losses, label="MAdamW")
    plt.plot(mradam_train_losses, label="MRAdam")
    plt.plot(mnadam_train_losses, label="MNAdam")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_save_path + "train_loss.png", dpi=300)
    # plt.show()


def plot_individual_results(results, figure_save_path):
    population_test_accuracies = results["population_test_accuracies"]
    population_test_losses = results["population_test_losses"]

    num_epochs = len(population_test_accuracies)
    num_individuals = len(population_test_accuracies[0])

    # 绘制每个个体的测试精度变化
    plt.figure(figsize=(9, 6))
    for i in range(num_individuals):
        accuracies = [population_test_accuracies[epoch][i] for epoch in range(num_epochs)]
        plt.plot(accuracies, label=f"Individual {i + 1}")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    # plt.title("Test Accuracy vs. Epoch for Each Individual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_save_path + "AGDE_test_acc.png", dpi=300)
    # plt.show()

    # 绘制每个个体的测试损失变化
    plt.figure(figsize=(9, 6))
    for i in range(num_individuals):
        losses = [population_test_losses[epoch][i] for epoch in range(num_epochs)]
        plt.plot(losses, label=f"Individual {i + 1}")
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    # plt.title("Test Loss vs. Epoch for Each Individual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_save_path + "AGDE_test_loss.png", dpi=300)
    # plt.show()

def plot_running_times(figure_save_path):
    # 加载所有 .pkl 文件
    agde_results = load_results(figure_save_path, "AGDE.pkl")
    msgd_results = load_results(figure_save_path, "MSGD.pkl")
    madam_results = load_results(figure_save_path, "MAdam.pkl")
    mrmsprop_results = load_results(figure_save_path, "MRMSprop.pkl")
    madamw_results = load_results(figure_save_path, "MAdamW.pkl")
    mradam_results = load_results(figure_save_path, "MRAdam.pkl")
    mnadam_results = load_results(figure_save_path, "MNAdam.pkl")

    agde_time = agde_results["total_time"]
    msgd_time = msgd_results["total_time"]
    madam_time = madam_results["total_time"]
    mrmsprop_time = mrmsprop_results["total_time"]
    madamw_time = madamw_results["total_time"]
    mradam_time = mradam_results["total_time"]
    mnadam_time = mnadam_results["total_time"]

    algorithms = ["AGDE", "MSGD", "MAdam", "MRMSprop", "MAdamW", "MRAdam", "MNAdam"]
    times = [agde_time, msgd_time, madam_time, mrmsprop_time, madamw_time, mradam_time, mnadam_time]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    plt.figure(figsize=(9, 6))
    plt.bar(algorithms, times, color=colors)
    plt.xlabel("Algorithm")
    plt.ylabel("Running Time (seconds)")
    plt.savefig(figure_save_path + "running_time.png", dpi=300)
    # plt.show()




figure_save_path = "./Results/STL10/"
agde_results = load_results(figure_save_path, "AGDE.pkl")
msgd_results = load_results(figure_save_path, "MSGD.pkl")
madam_results = load_results(figure_save_path, "MAdam.pkl")
mrmsprop_results = load_results(figure_save_path, "MRMSprop.pkl")
madamw_results = load_results(figure_save_path, "MAdamW.pkl")
mradam_results = load_results(figure_save_path, "MRAdam.pkl")
mnadam_results = load_results(figure_save_path, "MNAdam.pkl")

plot_results(agde_results, msgd_results, madam_results, mrmsprop_results, madamw_results,
             mradam_results, mnadam_results, figure_save_path)

plot_running_times(figure_save_path)
plot_individual_results(agde_results, figure_save_path)
