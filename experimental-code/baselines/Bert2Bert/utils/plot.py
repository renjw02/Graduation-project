import matplotlib.pyplot as plt

def plot_loss(loss_values):
    """
    绘制损失值随着训练迭代次数的变化曲线
    Args:
    - loss_values: list, 包含训练过程中每个迭代的损失值
    """
    plt.plot(loss_values, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()