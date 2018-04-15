import matplotlib.pyplot as plt


def plot_embeddings(embeddings, style='seaborn-white', title=None):
    plt.style.use(style)

    fig, ax = plt.subplots()

    for i, (x, y) in enumerate(embeddings):
        ax.scatter(x, y, color='purple')
        ax.annotate(i, xy=(x, y), fontsize=20)

    if title is not None:
        ax.set_title(title, fontsize=30)

    return fig, ax
