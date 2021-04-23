import os
import sys
from pathlib import Path
path = Path(os.getcwd()).parent
sys.path.append(str(path))
import config
from data_loader import TRAIN_loader, TEST_loader, preprocess_for_glow
import torchvision.datasets as dset
import torchvision.transforms as transforms

import argparse
import json
import shutil
import random
from itertools import islice

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from model import Glow

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage, Loss



def check_manual_seed(seed):
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    print("Using seed: {seed}".format(seed=seed))


def compute_loss(nll, reduction="mean"):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "none":
        losses = {"nll": nll}

    losses["total_loss"] = losses["nll"]

    return losses


def compute_loss_y(nll, y_logits, y_weight, y, multi_class, reduction="mean"):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "none":
        losses = {"nll": nll}

    if multi_class:
        y_logits = torch.sigmoid(y_logits)
        loss_classes = F.binary_cross_entropy_with_logits(
            y_logits, y, reduction=reduction
        )
    else:
        loss_classes = F.cross_entropy(
            y_logits, torch.argmax(y, dim=1), reduction=reduction
        )

    losses["loss_classes"] = loss_classes
    losses["total_loss"] = losses["nll"] + y_weight * loss_classes

    return losses


def main(
    train_dist,
    dataroot,
    modelroot,
    resume_train,
    download,
    augment,
    train_batchsize,
    eval_batchsize,
    epochs,
    seed,
    hidden_channels,
    K,
    L,
    actnorm_scale,
    flow_permutation,
    flow_coupling,
    LU_decomposed,
    learn_top,
    y_condition,
    y_weight,
    max_grad_clip,
    max_grad_norm,
    lr,
    workers,
    cuda,
    n_init_batches,
    output_dir,
    saved_optimizer,
    warmup,
    imageSize,
    train_loader,
    test_loader,
):
    
    batch_size = train_batchsize
    n_workers = workers

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    check_manual_seed(seed)

    #train_loader = TRAIN_loader(option=train_dist, is_glow=True)
    #test_loader = TEST_loader(train_dist=train_dist, target_dist=train_dist, batch_size=eval_batchsize, is_glow=True)
    
    if train_dist == "cifar10":
        image_shape = (32, 32, 3)
        num_classes = 10
    elif train_dist == "fmnist":
        image_shape = (28, 28, 1)
        num_classes = 10
        

    
    
    model = Glow(
        image_shape,
        hidden_channels,
        K,
        L,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
        num_classes,
        learn_top,
        y_condition,
    )

    model = model.to(device)
    optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=5e-5)

    lr_lambda = lambda epoch: min(1.0, (epoch + 1) / warmup)  # noqa
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def step(engine, batch):
        model.train()
        optimizer.zero_grad()

        x, y = batch
        x = x.to(device)

        if y_condition:
            y = y.to(device)
            z, nll, y_logits = model(x, y)
            losses = compute_loss_y(nll, y_logits, y_weight, y, multi_class)
        else:
            z, nll, y_logits = model(x, None)
            losses = compute_loss(nll)

        losses["total_loss"].backward()

        if max_grad_clip > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), max_grad_clip)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        return losses

    def eval_step(engine, batch):
        model.eval()

        x, y = batch
        x = x.to(device)

        with torch.no_grad():
            if y_condition:
                y = y.to(device)
                z, nll, y_logits = model(x, y)
                losses = compute_loss_y(
                    nll, y_logits, y_weight, y, multi_class, reduction="none"
                )
            else:
                z, nll, y_logits = model(x, None)
                losses = compute_loss(nll, reduction="none")

        return losses

    trainer = Engine(step)
    checkpoint_handler = ModelCheckpoint(
        output_dir, "glow", n_saved=2, require_empty=False
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        checkpoint_handler,
        {"model": model, "optimizer": optimizer},
    )

    monitoring_metrics = ["total_loss"]
    RunningAverage(output_transform=lambda x: x["total_loss"]).attach(
        trainer, "total_loss"
    )

    evaluator = Engine(eval_step)

    # Note: replace by https://github.com/pytorch/ignite/pull/524 when released
    Loss(
        lambda x, y: torch.mean(x),
        output_transform=lambda x: (
            x["total_loss"],
            torch.empty(x["total_loss"].shape[0]),
        ),
    ).attach(evaluator, "total_loss")

    if y_condition:
        monitoring_metrics.extend(["nll"])
        RunningAverage(output_transform=lambda x: x["nll"]).attach(trainer, "nll")

        # Note: replace by https://github.com/pytorch/ignite/pull/524 when released
        Loss(
            lambda x, y: torch.mean(x),
            output_transform=lambda x: (x["nll"], torch.empty(x["nll"].shape[0])),
        ).attach(evaluator, "nll")

    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)

    # load pre-trained model if one chose 'resume_train'
    if resume_train:
        filename = input(f'What is filename? (in GLOW_{train_dist} dir): ')
        model.load_state_dict(torch.load(f'../../{modelroot}/GLOW_{train_dist}/{filename}.pt'))
        model.set_actnorm_init()

        if saved_optimizer:
            optimizer.load_state_dict(torch.load(saved_optimizer))

        file_name, ext = os.path.splitext(f'../../{modelroot}/GLOW_{train_dist}/{filename}.pt')
        resume_epoch = int(file_name.split("_")[-1])

        @trainer.on(Events.STARTED)
        def resume_training(engine):
            engine.state.epoch = resume_epoch
            engine.state.iteration = resume_epoch * len(engine.state.dataloader)

    @trainer.on(Events.STARTED)
    def init(engine):
        model.train()

        init_batches = []
        init_targets = []

        with torch.no_grad():
            for batch, target in islice(train_loader, None, n_init_batches):
                init_batches.append(batch)
                init_targets.append(target)

            init_batches = torch.cat(init_batches).to(device)

            assert init_batches.shape[0] == n_init_batches * batch_size

            if y_condition:
                init_targets = torch.cat(init_targets).to(device)
            else:
                init_targets = None

            model(init_batches, init_targets)

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate(engine):
        evaluator.run(test_loader)

        scheduler.step()
        metrics = evaluator.state.metrics

        losses = ", ".join([f"{key}: {value:.2f}" for key, value in metrics.items()])

        print(f"Validation Results - Epoch: {engine.state.epoch} {losses}")

    timer = Timer(average=True)
    timer.attach(
        trainer,
        start=Events.EPOCH_STARTED,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED,
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        pbar.log_message(
            f"Epoch {engine.state.epoch} done. Time per batch: {timer.value():.3f}[s]"
        )
        timer.reset()

    trainer.run(train_loader, epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_dist",
        type=str,
        default="cifar10",
        choices=["cifar10", "fmnist"],
        help="Type of the dataset to be used.",
    )    
    
    args = parser.parse_args()
    
    if args.train_dist == "cifar10":
        opt = config.GLOW_cifar10
        kwargs = {x: getattr(opt, x) for x in vars(opt) if x[0] != "_"}
        print("learning for cifar10")
    
    elif args.train_dist == "fmnist":
        opt = config.GLOW_fmnist
        kwargs = {x: getattr(opt, x) for x in vars(opt) if x[0] != "_"}        
        print("learning for fmnist")
    
    """
    try:
        output_dir = os.path.abspath('../../' + kwargs['output_dir'])
        os.makedirs(output_dir)
    except FileExistsError:
        raise FileExistsError(
            "Please provide a path to a non-existing or empty output directory."  # noqa
        )
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dist', type=str, default='cifar10')
    parser.add_argument('--dataroot', type=str, default='../../../data')
    parser.add_argument('--modelroot', type=str, default='../../../saved_models')
    parser.add_argument('--resume_train', type=bool, default=False)
    parser.add_argument('--download', type=bool, default=False)
    parser.add_argument('--augment', type=bool, default=True)
    parser.add_argument('--train_batchsize', type=int, default=32)
    parser.add_argument('--eval_batchsize', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_channels', type=int, default=400)
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--L', type=int, default=3)
    parser.add_argument('--actnorm_scale', type=float, default=1.0)
    parser.add_argument('--flow_permutation', type=str, default='invconv')
    parser.add_argument('--flow_coupling', type=str, default='affine')
    parser.add_argument('--LU_decomposed', type=bool, default=True)
    parser.add_argument('--learn_top', type=bool, default=True)
    parser.add_argument('--y_condition', type=bool, default=False)
    parser.add_argument('--y_weight', type=float, default=0.01)
    parser.add_argument('--max_grad_clip', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=int, default=0)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--n_init_batches', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='../../../saved_models/GLOW_cifar10')
    parser.add_argument('--saved_optimizer', type=str, default='')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--imageSize', type=int, default=32)

    opt = parser.parse_args()
    
    preprocess = [preprocess_for_glow]
    
    if opt.train_dist == 'cifar10':
        trainset = dset.CIFAR10(
            root=opt.dataroot,
            download=True,
            train=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((opt.imageSize, opt.imageSize)),
                transforms.ToTensor(),
            ] + preprocess)
        )
        testset = dset.CIFAR10(
            root=opt.dataroot,
            download=True,
            train=False,
            transform=transforms.Compose([
                transforms.Resize((opt.imageSize)),
                transforms.ToTensor(),
            ] + preprocess),
        )
        opt.train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=opt.train_batchsize,
            shuffle=True,
            num_workers=int(opt.workers),
        )
        opt.test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=opt.eval_batchsize,
            shuffle=False,
            num_workers=int(opt.workers),
        )
        
    elif opt.train_dist == 'fmnist':
        opt.hidden_channels = 200
        opt.output_dir = '../../../saved_models/GLOW_fmnist'
        trainset = dset.FashionMNIST(
            root=opt.dataroot,
            download=True,
            train=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((opt.imageSize, opt.imageSize)),
                transforms.ToTensor(),
            ] + preprocess)
        )
        testset = dset.FashionMNIST(
            root=opt.dataroot,
            download=True,
            train=False,
            transform=transforms.Compose([
                transforms.Resize((opt.imageSize)),
                transforms.ToTensor(),
            ] + preprocess),
        )
        opt.train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=opt.train_batchsize,
            shuffle=True,
            num_workers=int(opt.workers)
        )
        opt.test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=opt.eval_batchsize,
            shuffle=False,
            num_workers=int(opt.workers),
        )
    else:
        raise NotImplementedError('Oops! Please insert 1 or 2. Bye~')
        
    name = opt.output_dir.split('/')[-1] # This is 'GLOW_cifar10' or 'GLOW_fmnist'
    print(f'Train Loader for {name} is ready !')
    print(f'Test Loader for {name} is ready !')
    print(f'Please see the path "{opt.output_dir}" for the saved model !')
    
    kwargs = {x: getattr(opt, x) for x in vars(opt) if x[0] != "_"}
    
    main(**kwargs)
    
    
    
