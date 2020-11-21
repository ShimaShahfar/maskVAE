import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.networks as networks
from collections import OrderedDict
from torch.utils import data
from data.vae import Cityscapes
from torchvision import utils, transforms
from torch.utils.data import DataLoader
# from torchvision.transforms import ToTensor, ToPILImage, Resize
from options.train_options import TrainOptions
import torch.distributed as dist
from distributed import (get_rank, synchronize, reduce_loss_dict, reduce_sum, get_world_size,)


def save_sample(mask, x, batch_id, mapping):
    rev_mapping = {mapping[k]: k for k in mapping}
    pred = torch.argmax(x, dim=1) # or e.g. pred = torch.randint(0, 19, (224, 224))
    pred_image = torch.zeros(3, pred.shape[1], pred.shape[2], dtype=torch.uint8)
    for i in range(pred.shape[0]):
        for k in rev_mapping:
            pred_image[:, pred[i]==k] = torch.tensor(rev_mapping[k]).byte().view(3, 1)
        sample = pred_image.float()
        with torch.no_grad():
            utils.save_image(
                sample,
                f"maskVAE_dist/{str(batch_id).zfill(6)}_{str(i)}.png",
                normalize=True,
                range = (0,255),
            )
            if batch_id == 0:
                rev_mapping = {mapping[k]: k for k in mapping}
                pred = torch.argmax(mask, dim=1) # or e.g. pred = torch.randint(0, 19, (224, 224))
                pred_image = torch.zeros(3, pred.shape[1], pred.shape[2], dtype=torch.uint8)
                for i in range(pred.shape[0]):
                    for k in rev_mapping:
                        pred_image[:, pred[i]==k] = torch.tensor(rev_mapping[k]).byte().view(3, 1)
                    ground_truth = pred_image.float()
                utils.save_image(
                    ground_truth,
                    f"maskVAE_dist/gt{str(batch_id).zfill(6)}_{str(i)}.png",
                    normalize=True,
                    range = (0,255),
                )

def vae_loss(logits, target_onehot, mu, logvar, cur_epoch, name="train"):
    pred = torch.argmax(logits, dim=1).float()
    target = torch.argmax(target_onehot, dim=1)
    recon_loss = F.cross_entropy(logits, target)

    acc = (pred == target).float().mean()

    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    print(
        (
            f"{name}\n"
            f"epoch #{cur_epoch}\n"
            f"reconstruction loss: {recon_loss :.5f}\n"
            f"kldivergence:  {kldivergence: .2f}\n"
            f"variational_beta * kldivergence: {variational_beta * kldivergence: .5f}\n"
            f"batch accuracy: {acc * 100:.2f}\n\n"
        )
    )

    return recon_loss, variational_beta * kldivergence

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)

variational_beta = 0.0001
num_epochs = 150
latent_dims = 1024
batch_size = 8
learning_rate = 1e-5
opt = TrainOptions().parse()

n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
opt.distributed = n_gpu > 1
print("opt.distributed: ", opt.distributed)
if opt.distributed:
    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    synchronize()

mapping = {(0, 0, 0):0, (0,  0,  0):1, (0,  0,  0):2, (0,  0,  0):3, (0,  0,  0):4, (111, 74,  0):5, (81, 0, 81):6,
            (128, 64,128):7, (244, 35,232):8, (250,170,160):9, (230,150,140):10, ( 70, 70, 70):11, (102,102,156):12, (190,153,153):13,
            (180,165,180):14, (150,100,100):15, (150,120,90):16, (153,153,153):17, (153,153,153):18, (250,170,30):19, (220,220,0):20,
            (107,142, 35):21, (152,251,152):22, ( 70,130,180):23, (220, 20, 60):24, (255,  0, 0):25, ( 0, 0, 142):26, ( 0, 0, 70):27,
            (  0, 60,100):28, (  0,  0, 90):29, (  0,  0,110):30, (  0, 80,100):31, ( 0,  0,230):32, (119, 11,32):33, ( 0, 0,142):34, 
            }

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netVAE = networks.define_VAE(input_nc=opt.input_nc, gpu_ids = opt.gpu_ids)
netVAE = netVAE.to(device)

dataset = Cityscapes(root=opt.dataroot, split='train', mode='fine', target_type='semantic', transform=None, target_transform=None, transforms=None)
dataset_val = Cityscapes(root=opt.dataroot, split='val', mode='fine', target_type='semantic', transform=None, target_transform=None, transforms=None)

dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=data_sampler(dataset, shuffle=True, distributed=opt.distributed),
            drop_last=True,
            )

dataloader_val = DataLoader(
            dataset_val,
            batch_size=batch_size,
            drop_last=True,
            sampler=data_sampler(dataset_val, shuffle=True, distributed=opt.distributed),
            )

if opt.distributed:
        netVAE = nn.parallel.DistributedDataParallel(
            netVAE,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
        )


# dataset = data_loader.load_data()
num_params = sum(p.numel() for p in netVAE.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)

PATH = "checkpoint_vae/000100.pt"
checkpoint = torch.load(PATH)
netVAE.load_state_dict(checkpoint['vae'])

optimizer = torch.optim.Adam(params=netVAE.parameters(), lr=learning_rate, weight_decay=1e-5)

train_loss_avg = []
val_loss_avg = []
train_loss_avg.append(0)
val_loss_avg.append(0)

print('Training ...')
for epoch in range(num_epochs):
    num_batches, n_iter = 0, 0
    for batch in dataloader:
        netVAE.train()
        batch = batch.to(device)
        batch = F.one_hot(batch.long(), num_classes=34)
        batch = batch.permute(0, 3, 1, 2).float().contiguous()

        logits, target_onehot, latent_mu, latent_logvar = netVAE(batch)
        losses = vae_loss(
            logits, target_onehot, latent_mu, latent_logvar, name="train", cur_epoch=epoch
        )
        l1, l2 = losses
        loss = l1 + l2
        optimizer.zero_grad()
        loss.backward()
        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()
        num_batches += 1
        train_loss_avg[-1] += loss.item()

    for batch in dataloader_val:
        netVAE.eval()
        with torch.no_grad():
            batch = batch.to(device)
            batch = F.one_hot(batch.long(), num_classes=34)
            batch = batch.permute(0, 3, 1, 2).float().contiguous()

            logits, target_onehot, latent_mu, latent_logvar = netVAE(batch)
            losses = vae_loss(
                logits, target_onehot, latent_mu, latent_logvar, name="val", cur_epoch=epoch
            )
            l1, l2 = losses
            loss_val = l1 + l2
            val_loss_avg[-1] += loss_val.item()
            n_iter +=1
            
    train_loss_avg[-1] /= num_batches
    val_loss_avg[-1] /= n_iter
    print(f'epoch # {epoch} : train loss is {train_loss_avg} and validation loss is {val_loss_avg} ')
        
    if epoch % 3 == 0:
        save_sample(target_onehot, logits, epoch, mapping)
        print("saved samples")

    if epoch % 5 == 0:
        torch.save(
            {
                "vae": netVAE.state_dict(), 
                "average_loss" : train_loss_avg,
                "validation_loss" : val_loss_avg,
                },
                f"checkpoint_vae_dist/{str(epoch).zfill(6)}.pt",
        )
    
    print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, num_epochs, train_loss_avg[-1]))

    