import csv
from datetime import datetime
from pathlib import Path

import cv2
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from loss import FocalLoss, BinaryDiceLoss
from tools import visualization, calculate_metric, calculate_average_metric
from .adaclip import *
from .custom_clip import create_model_and_transforms


class AdaCLIP_Trainer(nn.Module):
    def __init__(
            self,
            # clip-related
            backbone, feat_list, input_dim, output_dim,

            # learning-related
            learning_rate, device, image_size,

            # model settings
            prompting_depth=3, prompting_length=2,
            prompting_branch='VL', prompting_type='SD',
            use_hsf=True, k_clusters=20,
            anomaly_config=None,
    ):

        super(AdaCLIP_Trainer, self).__init__()

        self.device = device
        self.feat_list = feat_list
        self.image_size = image_size
        self.prompting_branch = prompting_branch
        self.prompting_type = prompting_type

        self.loss_focal = FocalLoss()
        self.loss_dice = BinaryDiceLoss()

        ########### different model choices
        freeze_clip, _, self.preprocess = create_model_and_transforms(backbone, image_size,
                                                                      pretrained='openai')
        freeze_clip  = freeze_clip.to(device)
        freeze_clip.eval()

        self.clip_model = Baseline(freeze_clip=freeze_clip,
                                  text_channel=output_dim,
                                  visual_channel=input_dim,
                                 prompting_length=prompting_length,
                                  prompting_depth=prompting_depth,
                                  prompting_branch=prompting_branch,
                                  prompting_type=prompting_type,
                                  use_hsf=use_hsf,
                                  k_clusters=k_clusters,
                                  output_layers=feat_list,
                                  device=device,
                                  image_size=image_size).to(device)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])

        self.preprocess.transforms[0] = transforms.Resize(size=(image_size, image_size),
                                                          interpolation=transforms.InterpolationMode.BICUBIC,
                                                          max_size=None)

        self.preprocess.transforms[1] = transforms.CenterCrop(size=(image_size, image_size))

        # update parameters
        self.learnable_paramter_list = [
            'text_prompter',
            'visual_prompter',
            'patch_token_layer',
            'cls_token_layer',
            'image_mapping',
            'fuse',
        ]

        self.params_to_update = []
        for name, param in self.clip_model.named_parameters():
            should_update = any(update_name in name for update_name in self.learnable_paramter_list)
            param.requires_grad_(should_update)
            if should_update:
                self.params_to_update.append(param)

        # build the optimizer
        seg_ids = {id(p) for n, p in self.clip_model.named_parameters() if 'seg_patch_token_layer' in n}
        self.optimizer = torch.optim.AdamW(
            [{'params': [p for p in self.params_to_update if id(p) not in seg_ids]},
            {'params': [p for p in self.params_to_update if id(p) in seg_ids], 'lr': 0.001}],
            lr=learning_rate, betas=(0.5, 0.999), weight_decay=0.01)

    def save(self, path):
        self.save_dict = {}
        for param, value in self.state_dict().items():
            for update_name in self.learnable_paramter_list:
                if update_name in param:
                    # print(f'{param}: {update_name}')
                    self.save_dict[param] = value
                    break

        torch.save(self.save_dict, path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device), strict=False)

    def train_one_batch(self, items):
        image = items['img'].to(self.device)
        cls_name = items['cls_name']

        # pixel level
        seg_maps, aux_maps, anomaly_score = self.clip_model(image, cls_name, aggregation=False)

        # if not isinstance(anomaly_map, list):
        #     anomaly_map = [anomaly_map]

        # losses
        gt = items['img_mask'].to(self.device)
        gt = gt.squeeze(1)

        gt[gt > 0.5] = 1
        gt[gt <= 0.5] = 0

        is_anomaly = items['anomaly'].to(self.device)
        is_anomaly[is_anomaly > 0.5] = 1
        is_anomaly[is_anomaly <= 0.5] = 0
        loss = 0

        # classification loss
        classification_loss = self.loss_focal(anomaly_score, is_anomaly.unsqueeze(1))
        loss += classification_loss

        # seg loss
        seg_l = lambda maps: sum(self.loss_focal(am, gt) + self.loss_dice(am[:, 1], gt)
                         + self.loss_dice(am[:, 0], 1 - gt) for am in maps)
        loss += seg_l(seg_maps) #+ seg_l(aux_maps)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def train_epoch(self, loader):
        self.clip_model.train()
        self.clip_model.freeze_clip.eval()
        loss_list = []
        for _, items in enumerate(tqdm(loader)):
            loss = self.train_one_batch(items)
            loss_list.append(loss.item())

        return np.mean(loss_list)

    @torch.no_grad()
    def evaluation(self, dataloader, obj_list, save_fig, save_fig_dir=None):
        self.clip_model.eval()
        # self.clip_model.dropout.train()  # enable dropout for evaluation

        # Only ONE class is kept in RAM at a time: its metrics are computed as soon as the class
        # ends, then the buffers are released. Requires samples grouped by class (shuffle=False),
        # which is how BaseDataset builds data_all.
        buffer_keys = ['cls_names', 'imgs_gts', 'anomaly_scores', 'imgs_masks', 'anomaly_maps', 'imgs', 'names']
        results = {k: [] for k in buffer_keys}
        class_metrics = {}
        finished_classes = set()
        current_cls = None
        score_rows = []
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

        def finish_class(cls):
            if save_fig:
                print(f'saving fig of {cls}.....')
                visualization.plot_sample_cv2(
                    results['names'],
                    results['imgs'],
                    {'AdaCLIP': results['anomaly_maps']},
                    results['imgs_masks'],
                    save_fig_dir
                )
            if cls in obj_list:
                class_metrics[cls] = calculate_metric(results, cls)
            finished_classes.add(cls)
            for k in buffer_keys:
                results[k].clear()

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_indx = 0
            for indx, items in enumerate(tqdm(dataloader)):
                image = items['img'].to(self.device)
                cls_name = items['cls_name']
                gt_mask = items['img_mask']
                gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0

                # pixel level
                anomaly_map, anomaly_score = self.clip_model(image, cls_name, aggregation=True)

                anomaly_map = anomaly_map.cpu().numpy()
                anomaly_score = anomaly_score.cpu().numpy()
                is_anomaly = np.array(items['anomaly'])

                # handled per sample so that a batch containing two classes is split correctly
                for i, _cls_name in enumerate(cls_name):
                    if _cls_name != current_cls:
                        if current_cls is not None:
                            finish_class(current_cls)
                        if _cls_name in finished_classes:
                            raise RuntimeError(
                                f"Class '{_cls_name}' appears again after it was already evaluated. "
                                f"evaluation() needs samples grouped by class (use shuffle=False).")
                        current_cls = _cls_name

                    if save_fig:
                        image_indx += 1
                        vis_image = cv2.resize(cv2.imread(items['img_path'][i]), (self.image_size, self.image_size))
                        results['imgs'].append(vis_image)
                        results['names'].append('{:}-{:03d}'.format(_cls_name, image_indx))

                    results['cls_names'].append(_cls_name)
                    # 0/1 mask stored as uint8 (4x smaller than float32); the copy also drops the batch tensor
                    results['imgs_masks'].append(gt_mask[i].squeeze(0).numpy().astype(np.uint8))  # px
                    results['anomaly_maps'].append(gaussian_filter(anomaly_map[i], sigma=4))
                    results['anomaly_scores'].append(anomaly_score[i])
                    results['imgs_gts'].append(is_anomaly[i])

        if current_cls is not None:
            finish_class(current_cls)

        # same key order as before: obj_list, then 'Average'; a class without samples gets zero metrics
        empty_results = {k: [] for k in buffer_keys}
        metric_dict = dict()
        for obj in obj_list:
            metric_dict[obj] = class_metrics[obj] if obj in class_metrics else calculate_metric(empty_results, obj)

        metric_dict['Average'] = calculate_average_metric(metric_dict)

        return metric_dict