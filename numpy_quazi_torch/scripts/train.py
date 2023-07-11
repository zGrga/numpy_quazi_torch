import argparse
from numpy_quazi_torch.custom_models.MNIST_model_2 import MNIST
from numpy_quazi_torch.models.SoftmaxLoss import SoftmaxLoss
from numpy_quazi_torch.data.fast.DatasetFast import DatasetFast
from numpy_quazi_torch.data.fast.DatasetIteratorFast import DatasetIteratorFast
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from numpy_quazi_torch.scheduler.ExponentialScheduler import ExponentialScheduler
import os
import uuid

def main():
    parser = argparse.ArgumentParser(description='Train model on MNIST dataset')
    parser.add_argument('--epochs', type=int, help='Number of epochs', required=True)
    # parser.add_argument('--lr', type=float, help='Learning rate [0.01]', required=False, default=0.05)
    parser.add_argument('--lr', type=float, help='Learning rate [0.01]', required=False, default=0.01)
    parser.add_argument('--dataset', type=str, help='Path to train dataset', required=True)
    parser.add_argument('--train_size', type=float, help='Train length [0.7]', default=0.7, required=False)
    parser.add_argument('--batch_size', type=int, help='Batch size', required=False, default=128)
    parser.add_argument('--store_plots', action='store_true', help='Store plots')
    parser.add_argument('--pretrained', type=str, required=False, help='Path to pretrained weights')

    args = parser.parse_args()

    if not os.path.exists('runs'):
        os.mkdir('runs')
    
    project_name = os.path.join('runs', str(uuid.uuid1()))
    os.makedirs(project_name)

    losses = []
    loss = SoftmaxLoss()
    model = MNIST()
    scheduler = ExponentialScheduler(
        start_lr=args.lr,
        stop_learning_rate=1e-8,
        step=0.95
    )

    if args.pretrained is not None:
        model.load_parameters(args.pretrained)
    
    dataset = DatasetFast(path=args.dataset)
    iterator = DatasetIteratorFast(
        dataset=dataset,
        batch_size=args.batch_size,
        train_flag=True,
        train_size=args.train_size
    )

    valid_loss, precision, recall, f1 = [], [], [], []
    lr = []

    last_f1 = None

    pbar = tqdm.tqdm(total=args.epochs, position=0)
    for i in range(args.epochs):
        iterator.to_train(True)
        iterator.reset_counters()

        pbar2 = tqdm.tqdm(total=len(iterator), position=1, leave=False)
        for index in range(len(iterator)):
            x, y = iterator.get()
            y_hat = model(x)

            loss_value = loss.loss(y_hat, y)
            losses.append(loss_value)

            pbar.set_description(f'Loss: {loss_value}')

            if args.store_plots:
                plt.figure()
                plt.plot(list(range(len(losses))), losses)
                plt.xlabel('Batch number')
                plt.ylabel('Loss')
                plt.title('Train loss on each batch')
                plt.savefig(os.path.join(project_name, 'batch_train_loss.png'))
                plt.close()

            to_ret = loss.backpropagation(loss(y_hat), y)
            model.backpropagation(to_ret, lr=scheduler())
            pbar2.update(1)
        
        lr.append(scheduler())

        iterator.to_train(False)

        global_y_argmax = []
        global_y_hat_argmax = []

        for index in range(len(iterator)):
            x, y = iterator.get()
            y_hat = model(x)

            valid_loss.append(loss.loss(y_hat, y))

            global_y_argmax.extend(np.argmax(y, axis=1).tolist())
            global_y_hat_argmax.extend(np.argmax(loss(y_hat), axis=1).tolist())

        tmp_precision, tmp_recall, tmp_f1, _ = precision_recall_fscore_support(global_y_argmax, global_y_hat_argmax, zero_division=0, average='macro')
        precision.append(tmp_precision)
        recall.append(tmp_recall)
        f1.append(tmp_f1)

        if last_f1 is None:
            last_f1 = f1
            model.store_parameters(os.path.join(project_name, 'weights'))
        elif last_f1 < f1:
            last_f1 = f1
            model.store_parameters(os.path.join(project_name, 'weights'))

        if args.store_plots:
            plt.figure()
            plt.plot(list(range(len(lr))), lr)
            plt.xlabel('Epoch')
            plt.ylabel('Learning rate')
            plt.title('Learning rate on each batch')
            plt.savefig(os.path.join(project_name, 'learning_rate.png'))
            plt.close()

            plt.figure()
            plt.plot(list(range(len(valid_loss))), valid_loss)
            plt.xlabel('Batch number')
            plt.ylabel('Loss')
            plt.title('Valid loss on each batch')
            plt.savefig(os.path.join(project_name, 'batch_valid_loss.png'))
            plt.close()


            plt.figure()
            plt.plot(list(range(len(precision))), precision, label='precision')
            plt.plot(list(range(len(recall))), recall, label='recall')
            plt.plot(list(range(len(f1))), f1, label='f1')
            plt.title('Metrics on each epoch [validation dataset]')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.legend()
            plt.savefig(os.path.join(project_name, 'metrics.png'))
        
        pbar2.refresh()
        scheduler.update()

        pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    main()