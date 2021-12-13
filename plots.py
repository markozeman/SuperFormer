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


def plot_multiple_results(num_tasks, num_epochs, first_average, means, stds, legend_lst, title, colors, x_label, y_label, vertical_lines_x, vl_min, vl_max, show_CI=True, text_strings=None):
    """
    Plot more lines from the saved results on the same plot with additional information.

    :param num_tasks: number of tasks trained
    :param num_epochs: number of epochs per task
    :param first_average: string - show results on 'first' task only or the 'average' results until current task index
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
    plt.grid(axis='y')

    # plot horizontal lines to explain learning
    for i in range(num_tasks):
        x_min = i * num_epochs if i == 0 else (i * num_epochs) - 1
        x_max = ((i + 1) * num_epochs) - 1
        plt.hlines(y=103, xmin=x_min, xmax=x_max)
        plt.vlines(x=x_min, ymin=102, ymax=104)
        plt.vlines(x=x_max, ymin=102, ymax=104)

        plt.text(x=x_min + 1, y=104, fontsize=12,
                 s='Learning %s; Results for %s' % (str(i+1), str(1) if first_average == 'first' else '1-%s' % (i+1) if i != 0 else str(1)))

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
        plt.legend(legend_lst, loc='lower left')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.vlines(vertical_lines_x, vl_min, vl_max, colors='k', linestyles='dashed', linewidth=2, alpha=0.5)
    if text_strings is not None:
        for i in range(len(text_strings)):
            plt.text(vertical_lines_x[i] + 0.5, vl_min, text_strings[i], color='k', alpha=0.5)
    plt.show()



