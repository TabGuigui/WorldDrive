import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
import imageio
import os
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import inf
else:
    from torch import inf

def init_optimizer(model, lr=1e-4, weight_decay=1e-3):
    optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    return optim


def init_lr_schedule(optimizer, milstones=[1000000, 1500000, 2000000], gamma=0.5):
    scheduler = MultiStepLR(optimizer, milestones=milstones, gamma=gamma)
    return scheduler

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        if args.lr_schedule == "constant":
            lr = args.lr
        elif args.lr_schedule == "cosine":
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        else:
            raise NotImplementedError
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler("cuda")

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list or 'diffloss' in name:
            no_decay.append(param)  # no weight decay on bias, norm and diffloss
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def save_model(self, path, epoch, rank=0):
    if rank == 0:
        save
        torch.save(self.model.state_dict(),'{}/tvar_{}.pkl'.format(path, str(epoch)))  

# def load_ckpt(load_path, model, optimizer=None, scheduler=None, strict_match=True, loss_scaler=None):
#     """
#     Load the check point for resuming training or finetuning.
#     """
#     if os.path.isfile(load_path):
#         if main_process():
#             logger.info(f"Loading weight '{load_path}'")
#         checkpoint = torch.load(load_path, map_location="cpu")
#         ckpt_state_dict  = checkpoint['model_state_dict']
#         model.module.load_state_dict(ckpt_state_dict, strict=strict_match)

#         if optimizer is not None:
#             optimizer.load_state_dict(checkpoint['optimizer'])
#         if scheduler is not None:
#             scheduler.load_state_dict(checkpoint['scheduler'])
#         if loss_scaler is not None and 'scaler' in checkpoint:
#             scheduler.load_state_dict(checkpoint['scaler'])
#         del ckpt_state_dict
#         del checkpoint
#         if main_process():
#             logger.info(f"Successfully loaded weight: '{load_path}'")
#             if scheduler is not None and optimizer is not None:
#                 logger.info(f"Resume training from: '{load_path}'")
#     else:
#         if main_process():
#             raise RuntimeError(f"No weight found at '{load_path}'")
#     return model, optimizer, scheduler, loss_scaler


def save_ckpt(args, path, model, optimizer=None, scheduler=None, curr_iter=0, curr_epoch=None):
    """
    Save the model, optimizer, lr scheduler.
    """

    ckpt = dict(
        model_state_dict=model.state_dict(),
        # optimizer_state_dict=optimizer.state_dict(),
        # scheduler=scheduler.state_dict(),
    )
    
    ckpt_path = '{}/tvar_{}.pkl'.format(path, str(curr_iter))

    torch.save(ckpt, ckpt_path)

    print(f'#### Save model: {ckpt_path}')

def resume_ckpt(local_rank, args, model, optimizer=None):
    
    resume_load_path = '{}/tvar_{}.pkl'.format(args.save_model_path, str(args.resume_step))
    print(local_rank,": loading...... ", resume_load_path)
    ckpt_file = torch.load(resume_load_path, map_location="cpu")
    
    if 'optimizer_state_dict' in ckpt_file:
        if optimizer is not None:
            optimizer.load_state_dict(ckpt_file['optimizer_state_dict'])
        print(local_rank, f'Rank: {local_rank}, Successfully loaded optimizer from {resume_load_path}.')
    if 'model_state_dict' in ckpt_file:
        print('loaded weight, model.pose_emb.weight sum:', torch.sum(ckpt_file['model_state_dict']['pose_emb.weight']))
        # model.load_state_dict(ckpt_file['model_state_dict'], strict=True)
        model = load_parameters(model, ckpt_file)
        print(local_rank, f'Rank: {local_rank}, Successfully loaded model from {resume_load_path}.')
    else:
        # model.load_state_dict(ckpt_file, strict=False)
        model = load_parameters(model, ckpt_file)
        print(local_rank, f'Rank: {local_rank}, Successfully loaded model from {resume_load_path}.')
    return model, optimizer



def load_parameters_transformer(model, load_ckpt_file, skip_key=None):
    if 'model_state_dict' in load_ckpt_file:
        ckpt = load_ckpt_file['model_state_dict']
    else:
        ckpt = load_ckpt_file
    ckpt_state_dict = {}
    for key, val in ckpt.items():
        key = "transformer." + key
        if key not in model.state_dict():
            print(f"Unused: {key}")
            continue
        elif val.shape == model.state_dict()[key].shape:
            ckpt_state_dict[key] = val
        else:
            param = torch.zeros_like(model.state_dict()[key])
            old_param = val
            if old_param.size() > param.size():
                if param.ndimension() == 1:
                    param = old_param[:param.size(0)]
                elif param.ndimension() == 2:
                    param = old_param[:param.size(0), :param.size(1)]
                elif param.ndimension() == 3:
                    param = old_param[:param.size(0), :param.size(1), :param.size(2)]
                elif param.ndimension() == 4:
                    param = old_param[:param.size(0), :param.size(1), :param.size(2), :param.size(3)]
                else:
                    raise NotImplementedError(f"Unsupported parameter dimension: {param.ndimension()}")
            else:
                if param.ndimension() == 1:
                    param[:old_param.size(0)] = old_param
                elif param.ndimension() == 2:
                    param[:old_param.size(0), :old_param.size(1)] = old_param
                elif param.ndimension() == 3:
                    param[:old_param.size(0), :old_param.size(1), :old_param.size(2)] = old_param
                elif param.ndimension() == 4:
                    param[:old_param.size(0), :old_param.size(1), :old_param.size(2), :old_param.size(3)] = old_param
                else:
                    raise NotImplementedError(f"Unsupported parameter dimension: {param.ndimension()}")
            ckpt_state_dict[key] = param
            print(f"Load: Shape of ckpt's {key} is {val.shape}, but model's shape is {model.state_dict()[key].shape}")
    ckpt_keys = list(ckpt.keys())
    newparas_not_in_ckpt = set(list(model.state_dict().keys())).difference(["transformer." + key for key in ckpt_keys])
    # newparas_not_in_ckpt = set(list(model.state_dict().keys())).difference(list(ckpt.keys()))
    for key in newparas_not_in_ckpt:
        print(f"Unfound: {key}")
        ckpt_state_dict[key] = model.state_dict()[key]
    model.load_state_dict(ckpt_state_dict, strict=True)
    return model

def load_parameters(model, load_ckpt_file, skip_key=None):
    if 'model_state_dict' in load_ckpt_file:
        ckpt = load_ckpt_file['model_state_dict']
    else:
        ckpt = load_ckpt_file
    ckpt_state_dict = {}
    for key, val in ckpt.items():
        key = key.split(".", 1)[1]
        if skip_key is not None and skip_key in key:
            # print(f"Skip: {key}")
            ckpt_state_dict[key] = model.state_dict()[key]
            continue
        if key not in model.state_dict():
            # print(f"Unused: {key}")
            continue
        elif val.shape == model.state_dict()[key].shape:
            ckpt_state_dict[key] = val
        else:
            param = torch.zeros_like(model.state_dict()[key])
            old_param = val
            if old_param.size() > param.size():
                if param.ndimension() == 1:
                    param = old_param[:param.size(0)]
                elif param.ndimension() == 2:
                    param = old_param[:param.size(0), :param.size(1)]
                elif param.ndimension() == 3:
                    param = old_param[:param.size(0), :param.size(1), :param.size(2)]
                elif param.ndimension() == 4:
                    param = old_param[:param.size(0), :param.size(1), :param.size(2), :param.size(3)]
                else:
                    raise NotImplementedError(f"Unsupported parameter dimension: {param.ndimension()}")
            else:
                if param.ndimension() == 1:
                    param[:old_param.size(0)] = old_param
                elif param.ndimension() == 2:
                    param[:old_param.size(0), :old_param.size(1)] = old_param
                elif param.ndimension() == 3:
                    param[:old_param.size(0), :old_param.size(1), :old_param.size(2)] = old_param
                elif param.ndimension() == 4:
                    param[:old_param.size(0), :old_param.size(1), :old_param.size(2), :old_param.size(3)] = old_param
                else:
                    raise NotImplementedError(f"Unsupported parameter dimension: {param.ndimension()}")
            ckpt_state_dict[key] = param
            print(f"Load: Shape of ckpt's {key} is {val.shape}, but model's shape is {model.state_dict()[key].shape}")
    ckpt_keys = list(ckpt.keys())
    newparas_not_in_ckpt = set(list(model.state_dict().keys())).difference([key.split('.', 1)[1] for key in ckpt_keys])
    # newparas_not_in_ckpt = set(list(model.state_dict().keys())).difference(list(ckpt.keys()))
    for key in newparas_not_in_ckpt:
        # print(f"Unfound: {key}")
        ckpt_state_dict[key] = model.state_dict()[key]
    model.load_state_dict(ckpt_state_dict, strict=True)
    return model

def load_parameters_vae(model, load_ckpt_file, skip_key=None):
    if 'model_state_dict' in load_ckpt_file:
        ckpt = load_ckpt_file['model_state_dict']
    else:
        ckpt = load_ckpt_file
    ckpt_state_dict = {}
    for key, val in ckpt.items():
        if skip_key is not None and skip_key in key:
            print(f"Skip: {key}")
            ckpt_state_dict[key] = model.state_dict()[key]
            continue
        if key not in model.state_dict():
            print(f"Unused: {key}")
            continue
        elif val.shape == model.state_dict()[key].shape:
            ckpt_state_dict[key] = val
        else:
            param = torch.zeros_like(model.state_dict()[key])
            old_param = val
            if old_param.size() > param.size():
                if param.ndimension() == 1:
                    param = old_param[:param.size(0)]
                elif param.ndimension() == 2:
                    param = old_param[:param.size(0), :param.size(1)]
                elif param.ndimension() == 3:
                    param = old_param[:param.size(0), :param.size(1), :param.size(2)]
                elif param.ndimension() == 4:
                    param = old_param[:param.size(0), :param.size(1), :param.size(2), :param.size(3)]
                else:
                    raise NotImplementedError(f"Unsupported parameter dimension: {param.ndimension()}")
            else:
                if param.ndimension() == 1:
                    param[:old_param.size(0)] = old_param
                elif param.ndimension() == 2:
                    param[:old_param.size(0), :old_param.size(1)] = old_param
                elif param.ndimension() == 3:
                    param[:old_param.size(0), :old_param.size(1), :old_param.size(2)] = old_param
                elif param.ndimension() == 4:
                    param[:old_param.size(0), :old_param.size(1), :old_param.size(2), :old_param.size(3)] = old_param
                else:
                    raise NotImplementedError(f"Unsupported parameter dimension: {param.ndimension()}")
            ckpt_state_dict[key] = param
            print(f"Load: Shape of ckpt's {key} is {val.shape}, but model's shape is {model.state_dict()[key].shape}")
    newparas_not_in_ckpt = set(list(model.state_dict().keys())).difference(list(ckpt.keys()))
    for key in newparas_not_in_ckpt:
        print(f"Unfound: {key}")
        ckpt_state_dict[key] = model.state_dict()[key]
    model.load_state_dict(ckpt_state_dict, strict=True)
    return model


def save_ckpt_deepspeed(args, path, model, optimizer=None, scheduler=None, curr_iter=0, curr_epoch=None):
    """
    Save the model, optimizer, lr scheduler.
    """

    client_sd = dict(
        curr_iter=curr_iter,
    )
    torch.distributed.barrier()
    os.makedirs(path, exist_ok=True)
    ckpt_path = path
    print(f'#### Deepspeed, Save model to {ckpt_path}')
    model.save_checkpoint(os.path.abspath(ckpt_path), curr_iter, client_sd, save_latest=True) #
    # print(f'#### Save model: {ckpt_path}')


def load_from_deepspeed_ckpt(args, model):
    if args.load_from_deepspeed is not None:
        print('#### Before deepspeed load ckpt, img_projector.0.weight sum:', torch.sum(model.model.state_dict()['img_projector.0.weight']))
        load_path, client_sd = model.load_checkpoint(args.load_from_deepspeed, load_module_strict=False, load_module_only=True)
        if load_path is None or client_sd is None:
            if args.load_from_deepspeed.endswith("/"):
                args.load_from_deepspeed = args.load_from_deepspeed[:-1]
            tag = os.path.split(args.load_from_deepspeed)[-1]
            resume_raw_dir = os.path.dirname(args.load_from_deepspeed) 
            load_path, client_sd = model.load_checkpoint(resume_raw_dir, tag, load_module_strict=False, load_module_only=True)
        print('#### After deepspeed load ckpt, img_projector.0.weight sum:', torch.sum(model.model.state_dict()['img_projector.0.weight']))
    # TODO
    # if args.resume_step > 0: # args.resume_from_deepspeed is not None:
    #     print('#### Before deepspeed resume ckpt, img_projector.0.weight sum:', torch.sum(model.model.state_dict()['img_projector.0.weight']))
    #     resume_load_path = '{}/{}'.format(args.save_model_path, str(args.resume_step))
    #     load_path, client_sd = model.load_checkpoint(resume_load_path)
    #     if load_path is None or client_sd is None:
    #         if resume_load_path.endswith("/"):
    #             resume_load_path = resume_load_path[:-1]
    #         tag = os.path.split(resume_load_path)[-1]
    #         # resume_raw_dir = os.path.split(args.resume_from_deepspeed)[0]
    #         load_path, client_sd = model.load_checkpoint(args.save_model_path, tag)
    #     # args.resume_step = client_sd['curr_iter']
    #     print('#### After deepspeed resume, img_projector.0.weight sum:', torch.sum(model.model.state_dict()['img_projector.0.weight']))
    return model