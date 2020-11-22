import argparse
import pandas as pd
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--show', action='store_true')

args = parser.parse_args()

lambda_vals = [0.5, 1.0, 10.0]

'''
Plot acc, em, f1 of joint_san + classifier for different lambdas
'''
for i,col in enumerate(['dev_acc', 'dev_em', 'dev_f1']):
    plt.gcf().clear()
    plt.title(f' dev {col.split("_")[-1]} for Joint SAN + classifier')
    plt.xlabel('epochs')
    plt.gca().set_ylim([0,100])
    for lval in lambda_vals:
        data = pd.read_csv(f'results/jsc_results_{lval}.csv', index_col='epoch')
        plt.plot(data[col], label=f'l={lval}')
    plt.legend(loc='upper left')
    plt.savefig(f'plots/lambda-{col}.jpg')
    if args.show:
        plt.show()

'''plot training losses for each lambda value'''
for lval in lambda_vals:
    data = pd.read_csv(f'results/jsc_results_{lval}.csv', index_col='epoch')
    plt.gcf().clear()
    plt.xlabel('epochs')
    train_loss = data['train_loss_san'] + lval*data['train_loss_class']
    plt.plot(train_loss, label='total loss')
    plt.plot(data['train_loss_san'], label='san loss')
    plt.plot(data['train_loss_class'], label='classifier loss')
    plt.legend(loc='upper left')
    plt.title(f'Training loss for Joint SAN + classifier, lambda = {lval}')
    plt.savefig(f'plots/train_loss_{lval}.jpg')
    if args.show:
        plt.show()
