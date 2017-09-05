import numpy as np
import re
import click
from matplotlib import pylab as plt

@click.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True))
def main(files):
    TOP_5_FLAG = True
    Training_Accuracy_FLAG = False
    print(files)
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    #ax3 = ax1
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('loss')
    #ax3.set_xlabel('iteration')
    #ax3.set_ylabel('loss')
    ax2.set_ylabel('accuracy %')
    ax2.set_ylim([0, 100])
    ax2.grid(b=True, which='major', color='w', linestyle='-')
    ax2.grid(b=True, which='minor', color='w', linestyle='--')

    for i, log_file in enumerate(files):
        print(i)
        loss_iterations, losses,  loss_iterations_val, losses_val, accuracy_iterations_train, accuracies_train, accuracy_iterations, accuracies, accuracies_iteration_checkpoints_ind, accuracy_iterations_t5, accuracies_t5 = parse_log(log_file)
        disp_results(TOP_5_FLAG,Training_Accuracy_FLAG,fig, ax1, ax2, ax1,ax2,ax2, loss_iterations, losses,  loss_iterations_val, losses_val, \
                     accuracy_iterations_train, accuracies_train, accuracy_iterations, accuracies, accuracies_iteration_checkpoints_ind, accuracy_iterations_t5, accuracies_t5, color_ind=i)
    plt.show()

def parse_log(log_file):
    with open(log_file, 'r') as log_file:
        log = log_file.read()

#===TRAINING LOSS
    loss_pattern = r"Iteration (?P<iter_num>\d+).*, loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    losses = []
    loss_iterations = []
    for r in re.findall(loss_pattern, log):
        loss_iterations.append(int(r[0]))
        losses.append(float(r[1]))
    loss_iterations = np.array(loss_iterations)
    losses = np.array(losses)

    #t='line 1: no.=-13.56\nline 2: no.=13.26e-3'
    #pt=r"line (?P<n>\d+):.* no.=(?P<nm>[+-]?(\d+\.\d*)([eE][+-]?\d+)?)"
    #print(re.findall(pt,t))

# ===TRAINING ACCURACY top-1
    accuracy_pattern_train = r"Iteration (?P<iter_num>\d+), .*\n.*\n.*\n.*\n.* loss3/top-1 = (?P<accuracy>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    accuracies_train = []
    accuracy_iterations_train = []

    for r in re.findall(accuracy_pattern_train, log):
        iteration = int(r[0])
        accuracy = float(r[1]) * 100

        accuracy_iterations_train.append(iteration)
        accuracies_train.append(accuracy)

    accuracy_iterations_train = np.array(accuracy_iterations_train)
    accuracies_train = np.array(accuracies_train)


    #===VALIDATION LOSS
    loss_pattern_val_1 = r"Iteration (?P<iter_num>\d+), Testing net \(#0\)\n.* \(\* 0.3 = (?P<loss_val>[+-]?(\d+\.\d*)([eE][+-]?\d+)?) loss\)"
    loss_pattern_val_2 = r"Iteration (?P<iter_num>\d+), Testing net \(#0\)\n.*\n.*\n.*\n.* \(\* 0.3 = (?P<loss_val>[+-]?(\d+\.\d*)([eE][+-]?\d+)?) loss\)"
    loss_pattern_val_3 = r"Iteration (?P<iter_num>\d+), Testing net \(#0\)\n.*\n.*\n.*\n.*\n.*\n.*\n.* \(\* 1 = (?P<loss_val>[+-]?(\d+\.\d*)([eE][+-]?\d+)?) loss\)"

    loss_iterations_val = []
    losses_val = []
    losses_val_1 = []
    losses_val_2 = []
    losses_val_3 = []
    for r in re.findall(loss_pattern_val_1, log):
        loss_iterations_val.append(int(r[0]))
        losses_val_1.append(float(r[1]))

    losses_val_1 = np.array(losses_val_1)

    for r in re.findall(loss_pattern_val_2, log):
        losses_val_2.append(float(r[1]))
    losses_val_2 = np.array(losses_val_2)

    for r in re.findall(loss_pattern_val_3, log):
        losses_val_3.append(float(r[1]))
    losses_val_3 = np.array(losses_val_3)

    loss_iterations_val = np.array(loss_iterations_val)
    print('=================')
    print(loss_iterations_val/400)
    print(len(losses_val_1))
    print(len(losses_val_2))
    print(len(losses_val_3))

    losses_val = losses_val_1 + losses_val_2 + losses_val_3

#===VALIDATION ACCURACY top-1
    #accuracy_pattern = r"Iteration (?P<iter_num>\d+), Testing net \(#0\)\n.* label/top-1 = (?P<accuracy>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    accuracy_pattern = r"Iteration (?P<iter_num>\d+), Testing net \(#0\)\n.*\n.*\n.*\n.*\n.*\n.*\n.*\n.* loss3/top-1 = (?P<accuracy>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    accuracies = []
    accuracy_iterations = []
    accuracies_iteration_checkpoints_ind = []

    for r in re.findall(accuracy_pattern, log):
        iteration = int(r[0])
        accuracy = float(r[1]) * 100

        if iteration % 10000 == 0 and iteration > 0:
            accuracies_iteration_checkpoints_ind.append(len(accuracy_iterations))

        accuracy_iterations.append(iteration)
        accuracies.append(accuracy)

    accuracy_iterations = np.array(accuracy_iterations)
    accuracies = np.array(accuracies)
    print(len(accuracies))

# ===VALIDATION ACCURACY top-5
    # accuracy_pattern = r"Iteration (?P<iter_num>\d+), Testing net \(#0\)\n.* label/top-1 = (?P<accuracy>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    accuracy_pattern_t5 = r"Iteration (?P<iter_num>\d+), Testing net \(#0\)\n.*\n.*\n.*\n.*\n.*\n.*\n.*\n.*\n.* loss3/top-5 = (?P<accuracy>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    accuracies_t5 = []
    accuracy_iterations_t5 = []
    accuracies_iteration_checkpoints_ind = []

    for r in re.findall(accuracy_pattern_t5, log):
        iteration = int(r[0])
        accuracy = float(r[1]) * 100

        if iteration % 10000 == 0 and iteration > 0:
            accuracies_iteration_checkpoints_ind.append(len(accuracy_iterations_t5))

        accuracy_iterations_t5.append(iteration)
        accuracies_t5.append(accuracy)

    accuracy_iterations_t5 = np.array(accuracy_iterations_t5)
    accuracies_t5 = np.array(accuracies_t5)

    return loss_iterations, losses, loss_iterations_val, losses_val, accuracy_iterations_train, accuracies_train, accuracy_iterations, accuracies, accuracies_iteration_checkpoints_ind,accuracy_iterations_t5, accuracies_t5,


def disp_results(TOP_5_FLAG,Training_Accuracy_FLAG,fig, ax1, ax2, ax3, ax4,ax5, loss_iterations, losses, loss_iterations_val, losses_val, \
                 accuracy_iterations_train, accuracies_train, accuracy_iterations, accuracies, accuracies_iteration_checkpoints_ind, accuracy_iterations_t5, accuracies_t5, color_ind=0):
    modula = len(plt.rcParams['axes.color_cycle'])
    ax1.plot(loss_iterations, losses, color=plt.rcParams['axes.color_cycle'][(color_ind * 2 + 0) % modula], label='loss_train')
    ax3.plot(loss_iterations_val, losses_val, color=plt.rcParams['axes.color_cycle'][(color_ind * 2 + 1) % modula], label='loss_val')
    ax2.plot(accuracy_iterations, accuracies, plt.rcParams['axes.color_cycle'][(color_ind * 2 + 2) % modula])
    #ax2.plot(accuracy_iterations[accuracies_iteration_checkpoints_ind], accuracies[accuracies_iteration_checkpoints_ind], 'o', color=plt.rcParams['axes.color_cycle'][(color_ind * 2 + 1) % modula], label='accuracy')
    if TOP_5_FLAG == True:
        ax4.plot(accuracy_iterations_t5, accuracies_t5, plt.rcParams['axes.color_cycle'][(color_ind * 2 + 3) % modula])
    if Training_Accuracy_FLAG == True:
        ax5.plot(accuracy_iterations_train, accuracies_train, plt.rcParams['axes.color_cycle'][(color_ind * 2 +4) % modula])


if __name__ == '__main__':
    main()
