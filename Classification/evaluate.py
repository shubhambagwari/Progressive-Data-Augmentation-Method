#!/usr/bin/env python

import argparse
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import cfg

from fvcore.common.checkpoint import Checkpointer

from pytorch_image_classification import (
    apply_data_parallel_wrapper,
    create_dataloader,
    create_loss,
    create_model,
    create_scheduler,
    create_optimizer,
    get_default_config,
    update_config,
)
from pytorch_image_classification.utils import (
    AverageMeter,
    create_logger,
    get_rank,
)


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,default='configs/imagenet/vgg16.yaml')
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    update_config(config)
    config.freeze()
    return config


def evaluate(config, model, test_loader, loss_func, logger):
    device = torch.device(config.device)
    model.eval()
    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    start = time.time()
    pred_raw_all = []
    pred_prob_all = []
    pred_label_all = []

    pred_cnt = [0] * cfg.NUM_CLASSESS
    pred_correct_cnt = [0] * cfg.NUM_CLASSESS
    ###

    ###
    with torch.no_grad():
        for data, targets in tqdm.tqdm(test_loader):
            data = data.to(device)
            targets = targets.to(device)
            outputs = model(data)
            loss = loss_func(outputs, targets)
            pred_raw_all.append(outputs.cpu().numpy())
            pred_prob_all.append(F.softmax(outputs, dim=1).cpu().numpy())
            _, preds = torch.max(outputs, dim=1)
            ###
            cpu_target,cpu_pred_label = targets.cpu().numpy()[0],preds.cpu().numpy()[0]
            pred_cnt[cpu_target] += 1
            if cpu_target == cpu_pred_label:
                pred_correct_cnt[cpu_target] += 1
            ###
            pred_label_all.append(preds.cpu().numpy())
            loss_ = loss.item()
            correct_ = preds.eq(targets).sum().item()
            num = data.size(0)

            loss_meter.update(loss_, num)
            correct_meter.update(correct_, 1)

        accuracy = correct_meter.sum / len(test_loader.dataset)

        elapsed = time.time() - start
        logger.info(f'Elapsed {elapsed:.2f}')
        logger.info(f'Loss {loss_meter.avg:.4f} Accuracy {accuracy:.4f}')

    for i in range(cfg.NUM_CLASSESS):
        name = cfg.LABEL2NAME[i]
        if pred_cnt[i] == 0:
            acc = 0
        else:
            acc = pred_correct_cnt[i]/pred_cnt[i]
        print(name,acc)

    preds = np.concatenate(pred_raw_all)
    probs = np.concatenate(pred_prob_all)
    labels = np.concatenate(pred_label_all)
    return preds, probs, labels, loss_meter.avg, accuracy


def main():
    config = load_config()

    if config.test.output_dir is None:
        output_dir = pathlib.Path(config.test.checkpoint).parent
    else:
        output_dir = pathlib.Path(config.test.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    logger = create_logger(name=__name__, distributed_rank=get_rank())

    model = create_model(config)
    model = apply_data_parallel_wrapper(config, model)
    optimizer = create_optimizer(config, model)
    test_loader = create_dataloader(config, is_train=False)


    scheduler = create_scheduler(config,
                                 optimizer,
                                 steps_per_epoch=len(test_loader))
    checkpointer = Checkpointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir,
                                save_to_disk=get_rank() == 0)
    # checkpointer = Checkpointer(model,
    #                             optimizer=optimizer,
    #                             save_dir=output_dir,
    #                             save_to_disk=get_rank() == 0)
    checkpointer.load(config.test.checkpoint)


    _, test_loss = create_loss(config)

    preds, probs, labels, loss, acc = evaluate(config, model, test_loader,
                                               test_loss, logger)

    #print(probs,labels)
    output_path = output_dir / f'predictions.npz'
    np.savez(output_path,
             preds=preds,
             probs=probs,
             labels=labels,
             loss=loss,
             acc=acc)


if __name__ == '__main__':
    main()
