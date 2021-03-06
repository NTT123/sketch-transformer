"""Training script."""

import random
import tempfile
import time
from argparse import ArgumentParser
from collections import deque
from functools import partial

import torch
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

import dataset
import wandb
from dataset import DrawQuick, dq_collate
from generate import generate, plot_encoded_figure
from model import SketchModel, create_model


def _prepare_batch(x, device):
    x = torch.from_numpy(x)
    x = torch.transpose(x, 0, 1).long().to(device)
    inputs = x[:-1]
    targets = x[1:, :, 0]
    mask = inputs[..., 0] != 513
    return inputs, targets, mask


def _loss_func(it, model, device):
    batch = next(it)
    inputs, targets, mask = _prepare_batch(batch, device)
    y_hat = model(inputs)
    y_hat = torch.transpose(y_hat, 1, 2)
    loss = cross_entropy(y_hat, targets, reduction='none')
    loss = loss * mask.detach()
    loss = loss.sum() / mask.sum().detach()
    return loss


def _train(start_iteration, model, optimizer, device, train_dataloader, test_dataloader, args):
    train_loss = deque(maxlen=args.log_freq)
    test_loss = deque(maxlen=args.log_freq)
    model = model.to(device)
    start_time = time.perf_counter()
    test_iter = iter(test_dataloader)
    train_iter = iter(train_dataloader)
    loss_func = partial(_loss_func, model=model, device=device)
    oclr = OneCycleLR(optimizer, args.learning_rate, pct_start=0.01, total_steps=1_000_000,
                      cycle_momentum=False, last_epoch=start_iteration - 2)

    for iteration in range(start_iteration, 1 + args.num_training_steps):
        loss = loss_func(train_iter)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        oclr.step()
        train_loss.append(loss.detach())

        if iteration % (10 * args.log_freq) == 0:
            ckpt = f'checkpoint_{iteration:07d}.pt'
            print('Saving checkpoint', ckpt)
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args
            }, ckpt)

        if iteration % 20 == 0:
            with torch.no_grad():
                model.eval()
                test_loss.append(loss_func(test_iter).detach())
                model.train()

        if iteration % args.log_freq == 0:
            avg_train_loss = sum(train_loss).item() / len(train_loss)
            avg_test_loss = sum(test_loss).item() / len(test_loss)
            end_time = time.perf_counter()
            duration, start_time = end_time - start_time, end_time
            lr = oclr.get_last_lr()[0]
            with torch.no_grad():
                model.eval()
                cat = random.randrange(0, len(dataset.categories))
                sample = generate(model, device, cat)
                model.train()
            train_sample = next(train_iter)[0, :]
            test_sample = next(test_iter)[0, :]
            plot_encoded_figure(train_sample[:, 0].tolist(), train_sample[0, 2], 'train_sample.png')
            plot_encoded_figure(test_sample[:, 0].tolist(), test_sample[0, 2], 'test_sample.png')
            plot_encoded_figure(sample, cat, 'random_sample.png')
            print(
                f'Iteration {iteration:07d}  Train loss {avg_train_loss:.3f}  Test loss {avg_test_loss:.3f}  LR {lr:.3e}  Duration {duration:.3f}')
            if args.use_wandb:
                wandb.log({
                    'iteration': iteration,
                    'train loss': avg_train_loss,
                    'test loss': avg_test_loss,
                    'duration': duration,
                    'learning rate': lr,
                    'train sample': wandb.Image('train_sample.png'),
                    'test sample': wandb.Image('test_sample.png'),
                    'random sample': wandb.Image('random_sample.png'),
                })


def main():
    parser = ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--dropout-rate', default=0.1, type=float)  # underfitting :-(
    parser.add_argument('--embed-dim', default=256, type=int)
    parser.add_argument('--learning-rate', default=1e-3, type=float)
    parser.add_argument('--log-freq', default=1000, type=int)
    parser.add_argument('--num-att-heads', default=4, type=int)
    parser.add_argument('--num-sketch-classes', default=345, type=int)
    parser.add_argument('--num-training-steps', default=1_000_000, type=int)
    parser.add_argument('--num-transformer-layers', default=8, type=int)
    parser.add_argument('--on-memory-dataset', default=False, action='store_true')
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--use-wandb', default=False, action='store_true')
    parser.add_argument('--vocab-size', default=514, type=int)
    args = parser.parse_args()

    if args.use_wandb:
        wandb.init(project='transformer-sketch')
        wandb.config.update(args)

    if args.on_memory_dataset:
        data = dataset.load_data_to_memory()
    else:
        data = None
    train_dataset = DrawQuick(data=data, mode='train')
    test_dataset = DrawQuick(data=data, mode='test')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1, collate_fn=dq_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=1, collate_fn=dq_collate)

    model, device, optimizer = create_model(args)
    print(model)
    print(train_dataset)
    start_iteration = 1
    if args.resume is not None:
        print('Loading checkpoint', args.resume)
        dic = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(dic['model_state_dict'])
        optimizer.load_state_dict(dic['optimizer_state_dict'])
        start_iteration = dic['iteration'] + 1

    _train(start_iteration, model, optimizer, device, train_dataloader, test_dataloader, args)


if __name__ == "__main__":
    main()
