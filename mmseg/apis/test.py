import os.path as osp
import pickle
import shutil
import tempfile
import datetime

import cv2
import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

import numpy as np
import torch
import torch.nn as nn
from mmseg.models.backbones.prompt import *
from mmseg.models.backbones.lora import *

import torch.nn.functional as F
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

def update_ema_variables_cotta(ema_model, model, alpha_teacher, iteration=None):
    # Use the "true" average until the exponential average is more correct
    if iteration:
        alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)

    if True:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def update_ema_variables(ema_model, model, alpha_model, alpha_prompt, iteration=None):
    # Use the "true" average until the exponential average is more correct
    if iteration:
        alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)

    if True:
        for ema_param, (name, param) in zip(ema_model.parameters(), model.named_parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            if "prompt" in name:
                ema_param.data[:] = alpha_prompt * ema_param[:].data[:] + (1 - alpha_prompt) * param[:].data[:]
            else:
                ema_param.data[:] = alpha_model * ema_param[:].data[:] + (1 - alpha_model) * param[:].data[:]
    return ema_model

def np2tmp(array, temp_file_name=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.

    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False).name
    np.save(temp_file_name, array)
    return temp_file_name

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def single_gpu_prmr(args,
                    model,
                    diffusion_model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    anchor=None,
                    ema_model=None,
                    anchor_model=None):
    
    anchor_model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    param_list = []
    out_dir = "results/PRMR/"+str(datetime.datetime.now())

    for name, param in model.named_parameters():
        if param.requires_grad:
            param_list.append(param)
        else:
            param.requires_grad=False

    optimizer = torch.optim.Adam(param_list, lr=args.model_lr, betas=(0.9, 0.999))

    for i, data in enumerate(data_loader):
        model.eval()
        ema_model.eval()

        with torch.no_grad():

            data_one = {
                'img_metas': [data['img_metas'][4] for i in range(1)],
                'img': [data['img'][4] for i in range(1)]
            }

            domain_weight = anchor_model.module.dmr(img=data_one['img'][0].cuda(),
                                                                        img_metas=data_one['img_metas'][0], 
                                                                        weight={"ema_model":ema_model.module.state_dict(),
                                                                                "model":model.state_dict(),
                                                                                "optimizer":optimizer.state_dict()})

            if domain_weight:
                weight = torch.load(domain_weight, map_location="cpu")
                ema_model.module.load_state_dict(weight["ema_model"])
                model.load_state_dict(weight["model"])
                optimizer.load_state_dict(weight['optimizer'])

            # teacher output
            result, preds, seg_logits = ema_model(return_loss=False, **data)
            uncertainty = -np.sum(seg_logits * np.log(seg_logits + 1e-8), axis=0)
            result = diffusion_model.dpr(seg_logits, uncertainty, result[0], args.time_step, args.threshold)

            _, prob_anchor, _= anchor_model(return_loss=False, **data_one)
            mask = (prob_anchor[0] > 0.69).astype(np.int64) # 0.74 was the 5% quantile for cityscapes, therefore we use 0.69 here
            result = [(mask*preds[4][0] + (1.-mask)*result[0]).astype(np.int64)]

            weight = 1.

        if isinstance(result, list):
            if len(data['img'])==14:
                img_id = 4 #The default size without flip
            else:
                img_id = 0

            loss = model.forward(return_loss=True, img=data['img'][img_id], img_metas=data['img_metas'][img_id].data[0],
                                 gt_semantic_seg=torch.from_numpy(result[0]).cuda().unsqueeze(0).unsqueeze(0))

            torch.mean(weight*loss["decode.loss_seg"]).backward()
            optimizer.step()
            optimizer.zero_grad()

            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        ema_model = update_ema_variables_cotta(ema_model = ema_model, model = model, alpha_teacher=args.ema_rate)

        for nm, m  in model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    mask = (torch.rand(p.shape)<0.01).float().cuda()
                    with torch.no_grad():
                        p.data = anchor[f"{nm}.{npp}"] * mask + p * (1.-mask)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results



def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results with CPU."""
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    """Collect results with GPU."""
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results

import time
import cv2
def draw_map(x):
    tic=time.time()
    name = str(tic)
    H=1080 // 2
    W=1920 // 2
    x_visualize = x
    print('ssssssssssss',x.shape)

    x_visualize =  x_visualize[0]
    x_visualize = (((x_visualize - np.min(x_visualize))/(np.max(x_visualize)-np.min(x_visualize)))*255).astype(np.uint8) #归一化并映射到0-255的整数，方便伪彩色化
    savedir = './cotta/uncer/'
    import os
    if not os.path.exists(savedir+'dense'):
        os.mkdir(savedir+'dense')
    x_visualize = cv2.applyColorMap(x_visualize, cv2.COLORMAP_WINTER)  # 伪彩色处理
    cv2.imwrite(savedir+'dense/'+name+'.jpg',x_visualize) #保存可视化图像
