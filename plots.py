import matplotlib.pyplot as plt


def plot_histogram(data, x_label, y_label, bins):
    """
    Plot histogram.

    :param data: numbers in a list
    :param x_label: label on x axis
    :param y_label: label on y axis
    :param bins: number of bins in a histogram
    :return: None
    """
    plt.hist(data, density=False, bins=bins)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_multiple_results(means, stds, legend_lst, title, colors, x_label, y_label, vertical_lines_x, vl_min, vl_max, show_CI=True, text_strings=None):
    """
    Plot more lines from the saved results on the same plot with additional information.

    :param means: list - [mean_acc, mean_auroc, mean_auprc]
    :param stds: list - [std_acc, std_auroc, std_auprc]
    :param legend_lst: list of label values (len(legend_lst)=len(means))
    :param title: plot title (string)
    :param colors: list of colors used for lines (len(colors)=len(means))
    :param x_label: label of axis x (string)
    :param y_label: label of axis y (string)
    :param vertical_lines_x: x values of where to draw vertical lines
    :param vl_min: vertical lines minimum y value
    :param vl_max: vertical lines maximum y value
    :param show_CI: show confidence interval range (boolean)
    :param text_strings: optional list of text strings to add to the bottom of vertical lines
    :return: None
    """
    font = {'size': 18}
    plt.rc('font', **font)

    # plot lines with confidence intervals
    i = 0
    for mean, std in zip(means, stds):
        # plot the shaded range of the confidence intervals (mean +/- std)
        if show_CI:
            up_limit = mean + std
            up_limit[up_limit > 100] = 100  # cut accuracies above 100
            down_limit = mean - std
            plt.fill_between(range(len(std)), up_limit, down_limit, color=colors[i], alpha=0.25)

        # plot the mean on top
        plt.plot(range(len(mean)), mean, colors[i], linewidth=3)

        i += 1

    if legend_lst:
        plt.legend(legend_lst)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.vlines(vertical_lines_x, vl_min, vl_max, colors='k', linestyles='dashed', linewidth=2, alpha=0.5)
    if text_strings is not None:
        for i in range(len(text_strings)):
            plt.text(vertical_lines_x[i] + 0.5, vl_min, text_strings[i], color='k', alpha=0.5)
    plt.show()



