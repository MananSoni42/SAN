import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()
parser.add_argument('--show', action='store_true', help='display plots before saving them')
parser.add_argument('--clear', action='store_true', help='clear the plot/ directory')
parser.add_argument('--lambda_comp', action='store_true', help='Compare metrics for different lambda values')
parser.add_argument('--san_off', action='store_true', default=False,
                    help='Don\'t show results for joint san')
parser.add_argument('--san_class_off', action='store_true', default=False,
                    help='Don\'t show results for joint san + classifier')
args = parser.parse_args()

lambda_vals = [1, 1.5, 10]
datas = ['train', 'dev']

if args.clear:
    os.system('gio trash plots/*.jpg')

#### Joint SAN ####
if not args.san_off:
    if args.lambda_comp:
        # Plot em, f1 of joint_san for different lambdas
        for dataset in datas:
            for i,metric in enumerate([f'{dataset}_acc', f'{dataset}_em', f'{dataset}_f1']):
                plt.gcf().clear()
                plt.title(f' dev {metric.split("_")[-1]} for Joint SAN + classifier')
                plt.xlabel('epochs')
                plt.gca().set_ylim([0,100])
                for lval in lambda_vals:
                    data = pd.read_csv(f'results/js_results_{lval}.csv', index_col='epoch')
                    plt.plot(data[metric], label=f'l={lval}')
                plt.legend(loc='upper left')
                plt.savefig(f'plots/js_lambda-{dataset}-{metric}.jpg')
                if args.show:
                    plt.show()

    # plot training losses for each lambda value
    for lval in lambda_vals:
        data = pd.read_csv(f'results/js_results_{lval}.csv', index_col='epoch')
        plt.gcf().clear()
        plt.xlabel('epochs')
        train_loss = data['train_loss_san'] + lval*data['train_loss_class']
        plt.plot(train_loss, label='total loss')
        plt.plot(data['train_loss_san'], label='san loss')
        plt.plot(data['train_loss_class'], label='classifier loss')
        plt.legend(loc='upper left')
        plt.title(f'Training loss for Joint SAN + classifier, lambda = {lval}')
        plt.savefig(f'plots/js_train_loss_{lval}.jpg')
        if args.show:
            plt.show()


#### Joint SAN + Classifier ####
if not args.san_class_off:
    if args.lambda_comp:
        # Plot acc, em, f1 of joint_san + classifier for different lambdas
        for dataset in datas:
            for i,metric in enumerate([f'{dataset}_acc', f'{dataset}_em', f'{dataset}_f1']):
                plt.gcf().clear()
                plt.title(f' dev {metric.split("_")[-1]} for Joint SAN + classifier')
                plt.xlabel('epochs')
                plt.gca().set_ylim([0,100])
                for lval in lambda_vals:
                    data = pd.read_csv(f'results/jsc_results_{lval}.csv', index_col='epoch')
                    plt.plot(data[metric], label=f'l={lval}')
                plt.legend(loc='upper left')
                plt.savefig(f'plots/jsc_lambda-{dataset}-{metric}.jpg')
                if args.show:
                    plt.show()

    # plot training losses for each lambda value
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
        plt.savefig(f'plots/jsc_train_loss_{lval}.jpg')
        if args.show:
            plt.show()
