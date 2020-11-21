import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as plF

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import models.networks as networks
from util.image_pool import ImagePool


class LitMaskNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.cfg = config

        if self.cfg.resize_or_crop != "none" or self.cfg.isTrain is False:
            # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True

        mask_channels = (
            self.cfg.label_nc if self.cfg.label_nc != 0 else self.cfg.input_nc
        )

        # generator network
        self.netG = networks.define_G(
            mask_channels,
            self.cfg.output_nc,  # image channels
            self.cfg.ngf,  # gen filters in first conv layer
            self.cfg.netG,  # global or local
            self.cfg.n_downsample_global,  # num of downsampling layers in netG
            self.cfg.n_blocks_global,  # num of residual blocks
            self.cfg.n_local_enhancers,  # ignored
            self.cfg.n_blocks_local,  # ignored
            self.cfg.norm,  # instance normalization or batch normalization
        )

        # discriminator network
        if self.isTrain:
            use_sigmoid = self.cfg.lsgan is False
            netD_input_nc = mask_channels + self.cfg.output_nc
            self.netD = networks.define_D(
                netD_input_nc,
                self.cfg.ndf,  # filters in first conv layer
                self.cfg.n_layers_D,
                self.cfg.norm,
                use_sigmoid,
                self.cfg.num_D,
                getIntermFeat=self.cfg.ganFeat_loss,
            )

            netB_input_nc = self.cfg.output_nc * 2
            self.netB = networks.define_B(
                netB_input_nc, self.cfg.output_nc, 32, 3, 3, self.cfg.norm
            )

        # loss functions
        self.use_pool = self.cfg.pool_size > 0
        if self.cfg.pool_size > 0:
            self.fake_pool = ImagePool(self.cfg.pool_size)

        self.criterionGAN = networks.GANLoss(
            use_lsgan=self.cfg.lsgan, tensor=self.Tensor
        )
        self.criterionFeat = torch.nn.L1Loss()
        self.criterionVGG = networks.VGGLoss(self.gpu_ids)

    def forward(self, image_target, mask_target, mask_inter, mask_outer):
        # encode
        # ...

        # Stage 1
        fake_image_target = self.netG.forward(mask_target, mask_target, image_target)

        # Stage 2
        fake_image_inter = self.netG.forward(mask_inter, mask_target, image_target)
        fake_image_outer = self.netG.forward(mask_outer, mask_target, image_target)
        fake_image_blend, alpha = self.netB.forward(fake_image_inter, fake_image_outer)

        return (
            fake_image_target,
            fake_image_inter,
            fake_image_outer,
            fake_image_blend,
            alpha,
        )

    def _get_D_inputs(self, mask, image):
        inputs = torch.cat((mask, image.detach()), dim=1).detach()
        if self.use_pool:
            inputs = self.fake_pool.query(inputs)
        return inputs

    def discriminator_loss(
        self, image_target, mask_target, fake_image_target, fake_image_blend,
    ):
        # fake samples
        inputs_D_fake = self._get_D_inputs(mask_target, fake_image_target)
        pred_D_fake = self.netD(inputs_D_fake)
        loss_D_fake = self.criterionGAN(pred_D_fake, target_is_real=False)

        inputs_D_blend = self._get_D_inputs(mask_target, fake_image_blend)
        pred_D_blend = self.netD(inputs_D_blend)
        loss_D_blend = self.criterionGAN(pred_D_blend, target_is_real=False)

        # real samples
        inputs_D_real = self._get_D_inputs(mask_target, image_target)
        pred_D_real = self.netD(inputs_D_real)
        loss_D_real = self.criterionGAN(pred_D_real, target_is_real=True)

        return loss_D_fake, loss_D_blend, loss_D_real

    def generator_loss(
        self,
        image_target,
        mask_target,
        mask_inter,
        mask_outer,
        fake_image_target,
        fake_image_inter,
        fake_image_outer,
        fake_image_blend,
    ):
        inputs_D_fake = torch.cat((mask_target, fake_image_target), dim=1)
        pred_D_fake = self.netD(inputs_D_fake)
        loss_G_GAN = self.criterionGAN(pred_D_fake, target_is_real=True)

        inputs_D_blend = torch.cat((mask_target, fake_image_blend), dim=1)
        pred_D_blend = self.netD(inputs_D_blend)
        loss_GB_GAN = self.criterionGAN(pred_D_blend, target_is_real=True)

        inputs_D_real = torch.cat((mask_target, image_target), dim=1)
        pred_real = self.netD(inputs_D_real)
        pred_real = pred_real.detach()

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        loss_GB_GAN_Feat = 0
        if self.cfg.ganFeat_loss:
            feat_weights = 4.0 / (self.cfg.n_layers_D + 1)
            D_weights = 1.0 / self.cfg.num_D
            for i in range(self.cfg.num_D):
                for j in range(len(pred_D_fake[i]) - 1):
                    loss_G_GAN_Feat += (
                        D_weights
                        * feat_weights
                        * self.criterionFeat(pred_D_fake[i][j], pred_real[i][j])
                        * self.cfg.lambda_feat
                    )
                    loss_GB_GAN_Feat += (
                        D_weights
                        * feat_weights
                        * self.criterionFeat(pred_D_blend[i][j], pred_real[i][j])
                        * self.cfg.lambda_feat
                    )

        # VGG feature matching loss
        loss_G_VGG = 0
        loss_GB_VGG = 0
        if self.cfg.vgg_loss:
            loss_G_VGG += (
                self.criterionVGG(fake_image_target, image_target)
                * self.cfg.lambda_feat
            )
            loss_GB_VGG += (
                self.criterionVGG(fake_image_blend, image_target) * self.cfg.lambda_feat
            )

        return (
            loss_G_GAN,
            loss_GB_GAN,
            loss_G_GAN_Feat,
            loss_GB_GAN_Feat,
            loss_G_VGG,
            loss_GB_VGG,
        )

    def training_step(self, batch, batch_idx):
        (
            fake_image_target,
            fake_image_inter,
            fake_image_outer,
            fake_image_blend,
            alpha,
        ) = self.forward(*batch)

        (opt_g, opt_d) = self.optimizers()

        loss_g = self.generator_loss(...)
        self.manual_backward(loss_g, opt_g)
        self.manual_optimizer_step(opt_g)

        # do anything you want
        loss_d = self.discriminator_loss(...)
        self.manual_backward(loss_d, opt_d)
        self.manual_optimizer_step(opt_d)

        # logging

    def setup(self, stage):
        # train, val, test = load_datasets(self.dataset_path)

        self.train_dataset = train
        self.val_dataset = val
        self.test_dataset = test

    def configure_optimizers(self):
        # optimizer G + B
        optimizer_G = torch.optim.AdamW(
            list(self.netG.parameters()) + list(self.netB.parameters()),
            lr=self.cfg.lr,
            betas=(self.cfg.beta1, 0.999),
            weight_decay=self.cfg.weight_decay,
        )

        # optimizer D
        optimizer_D = torch.optim.AdamW(
            self.netD.parameters(),
            lr=self.cfg.lr,
            betas=(self.cfg.beta1, 0.999),
            weight_decay=self.cfg.weight_decay,
        )
        return optimizer_G, optimizer_D

    def training_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        inputs, targets = batch
        outputs = self.forward(inputs)

        loss = self.criterion(outputs, targets)
        logs = {
            f"{prefix}_{metric_name}": meter(outputs, targets)
            for metric_name, meter in self.metrics.items()
        }
        logs[f"{prefix}_loss"] = loss
        self.log_dict(logs, prog_bar=True, logger=True)

        results = {"loss": loss, "metrics": logs}
        return results

    # def _aggregate_results(self, outputs):
    #     metrics = outputs[0]["metrics"].keys()
    #     results = {
    #         m: torch.stack([x["metrics"][m] for x in outputs]).mean().item()
    #         for m in metrics
    #     }
    #     self.log_dict(results, on_epoch=True, prog_bar=True, logger=True)
    #     return results

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_wrokers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_wrokers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_wrokers,
        )


def m_wrapper(fn, apply, **kwargs):
    def wrapper(outputs, targets):
        predictions = apply(outputs, dim=1)
        return fn(predictions, targets, **kwargs)

    return wrapper


if __name__ == "__main__":
    experiment_name = "first-try"

    dataset_path = "dataset"
    num_classes = 3
    learning_rate = 5e-4
    batch_size = 64
    metrics = {
        "accuracy": m_wrapper(plF.classification.accuracy, torch.argmax),
        "weighted_accuracy": m_wrapper(
            plF.classification.accuracy, torch.argmax, class_reduction="weighted"
        ),
        "average_precision": plF.classification.average_precision,
        "f1_score": plF.classification.f1_score,
    }

    num_sanity_val_steps = 10
    max_epochs = 40

    num_gpus = torch.cuda.device_count()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"checkpoints/{experiment_name}",
        filename="masknet-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    logger = pl.loggers.TensorBoardLogger("logs", name=experiment_name)

    model = LitMaskNet(
        dataset_path=dataset_path,
        metrics=metrics,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        logger=logger,
        gpus=num_gpus,
        num_sanity_val_steps=num_sanity_val_steps,
        max_epochs=max_epochs,
        flush_logs_every_n_steps=10,
        progress_bar_refresh_rate=5,
        weights_summary="full",
    )

    trainer.fit(model)