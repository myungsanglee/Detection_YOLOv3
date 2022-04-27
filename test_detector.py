import os
import cv2
import random
import argparse
import numpy as np
import albumentations
import albumentations.pytorch

import torch
from torchsummary import summary
from models.detector.yolov3 import YOLOv3
from models.detector.anchors import pascal_voc
from module.detector import Detector
from utils.module_select import get_model
from utils.utility import preprocess_input
from utils.yaml_helper import get_train_configs
from utils.nms import non_max_suppression
from utils.metric import MeanAveragePrecision
from torch.utils.data import Dataset, DataLoader
from dataset.detection.utils import collater
from dataset.detection.yolo_format import YoloDataset


def parse_names(names_file):
    # names_file = os.getcwd()+names_file
    with open(names_file, 'r') as f:
        return f.read().splitlines()


def gen_random_colors(names):
    colors = [(random.randint(0, 255),
               random.randint(0, 255),
               random.randint(0, 255)) for i in range(len(names))]
    return colors


def decode_boxes(input, anchors, num_classes, image_size):
    num_anchors = len(anchors)
    bbox_attrs = 5+num_classes
    bs = input.size(0)
    in_w = input.size(2)
    in_h = input.size(3)
    stride_h = image_size[1] / in_h
    stride_w = image_size[0] / in_w
    scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in anchors]
    prediction = input.view(bs, num_anchors, bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
    
    # Get outputs
    x = torch.sigmoid(prediction[..., 0])          # Center x
    y = torch.sigmoid(prediction[..., 1])          # Center y
    w = prediction[..., 2]                         # Width
    h = prediction[..., 3]                         # Height
    conf = torch.sigmoid(prediction[..., 4])       # Conf
    pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.
    
    FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
    # Calculate offsets for each grid
    grid_x = torch.linspace(0, in_w-1, in_w).repeat(in_w, 1).repeat(
        bs * num_anchors, 1, 1).view(x.shape).type(FloatTensor)
    grid_y = torch.linspace(0, in_h-1, in_h).repeat(in_h, 1).t().repeat(
        bs * num_anchors, 1, 1).view(y.shape).type(FloatTensor)
    # Calculate anchor w, h
    anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
    anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
    anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
    anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
    # Add offset and scale with anchors
    pred_boxes = FloatTensor(prediction[..., :4].shape)
    pred_boxes[..., 0] = x.data + grid_x
    pred_boxes[..., 1] = y.data + grid_y
    pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
    pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
    # Results
    _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
    output = torch.cat((pred_boxes.view(bs, -1, 4) * _scale,
                        conf.view(bs, -1, 1), pred_cls.view(bs, -1, num_classes)), -1)
    return output.data


def main(cfg, image_name, save):
    input_size = cfg['input_size']
    test_transform = albumentations.Compose([
        albumentations.Resize(input_size, input_size, always_apply=True),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ], bbox_params=albumentations.BboxParams(format='yolo', min_visibility=0.1))

    data_loader = DataLoader(YoloDataset(
            transforms=test_transform,
            path=cfg['val_path']),
            batch_size=1,
            shuffle=False,
            collate_fn=collater
    )
    for sample in data_loader:
        batch_x = sample['img']
        batch_y = sample['annot']
        
        print(f'batch_x: {batch_x.size()}')
        print(f'batch_y: {batch_y.size()}')
        
        break
    
    names = parse_names(cfg['names'])
    colors = gen_random_colors(names)

    # Preprocess Image
    image = cv2.imread(image_name)
    image = cv2.resize(image, (416, 416))
    image_2 = image.copy()
    # image_inp = preprocess_input(image)
    # image_inp = image_inp.unsqueeze(0)
    image_inp = batch_x
    if torch.cuda.is_available:
        image_inp=image_inp.cuda()

    # Load trained model
    backbone = get_model(cfg['backbone'])
    model = YOLOv3(Backbone=backbone, num_classes=cfg['classes'], in_channels=cfg['in_channels'], varient=cfg['varient'])
    if torch.cuda.is_available:
        model = model.to('cuda')

    model_module = Detector.load_from_checkpoint(
        # '/home/insig/Detection_YOLOv3/saved/DarkNet53_YOLOv3_Pascal/version_0/checkpoints/last.ckpt', 
        './saved/DarkNet53_YOLOv3_Pascal/version_0/checkpoints/epoch=234-step=234.ckpt', 
        # './saved/DarkNet53_YOLOv3_Pascal/version_1/checkpoints/epoch=494-step=494.ckpt', 
        model=model
    )
    model_module.eval()
    preds = model_module(image_inp)
    decoded = []
    for i, pred in enumerate(preds):
        # decoded.append(decode_boxes(pred, pascal_voc['anchors'][i], pascal_voc['classes'], [cfg['input_size'], cfg['input_size']]))
        decoded.append(decode_boxes(pred, pascal_voc['anchors'][i], cfg['classes'], [cfg['input_size'], cfg['input_size']]))
    output = torch.cat(decoded, 1)
    batch_detections = non_max_suppression(output, num_classes=cfg['classes'], nms_thres=0.45)
    
    map_metric = MeanAveragePrecision(cfg['classes'], pascal_voc['anchors'], cfg['input_size'])
    map_metric.update_state(batch_y, preds)
    map = map_metric.result()
    print(f'mAP: {map}')
    print(f'{map_metric._input_size}')
    
    for i, detections in enumerate(batch_detections):
        if detections is not None:
            print(f'\n\n{detections.size()}\n\n')
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Rescale coordinates to original dimensions
                ori_h, ori_w, _ = image.shape
                pre_h, pre_w = [cfg['input_size'], cfg['input_size']]
                y1 = int((y1 / pre_h) * ori_h)
                x1 = int((x1 / pre_w) * ori_w)
                y2 = int((y2 / pre_h) * ori_h)
                x2 = int((x2 / pre_w) * ori_w)
                # Create a Rectangle patch
                image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
            cv2.imshow("dkdk", image)
            cv2.waitKey(0)
            # cv2.imwrite("./inference/result/inference.png", image)
    cv2.destroyAllWindows()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str,
                        help='Train config file')
    parser.add_argument('--save', action='store_true',
                        help='Train config file')

    args = parser.parse_args()
    cfg = get_train_configs(args.cfg)
    # main(cfg, './inference/sample/test.jpg', args.save)
    main(cfg, './data/test.jpg', args.save)