import joblib
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    filename = "./Results_AdaptiveGDE/lenet5/MNIST_1000/results_0.001_0.3_0.7_0.1_1_0.1.pkl"
    plot_curves(filename)

    plot_combined_curves(filename)