#!/usr/bin/env python3
"""
Debug RADM output format
"""
import torch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog

from RADM import add_radm_config, add_rl_config, RADMDatasetMapper
from RADM.util.model_ema import add_model_ema_configs
from detectron2.data import build_detection_train_loader


def setup_cfg():
    """Setup configuration"""
    cfg = get_cfg()
    add_radm_config(cfg)
    add_rl_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file('configs/radm.yaml')
    cfg.MODEL.DEVICE = 'cpu'
    cfg.freeze()
    return cfg


def main():
    cfg = setup_cfg()

    # Register layout dataset
    from detectron2.data.datasets.coco import load_coco_json
    DATASET_ROOT = cfg.DATASETS.DATASET_PATH
    ANN_ROOT = '/home/sxm/flux-workspace/text-to-layout-zhuanlan/RADM/dataSets/RADM_dataset/annotations'
    TRAIN_JSON = '/home/sxm/flux-workspace/text-to-layout-zhuanlan/RADM/dataSets/RADM_dataset/annotations/train.json'
    IMAGE_ROOT = '/home/sxm/flux-workspace/text-to-layout-zhuanlan/RADM/dataSets/RADM_dataset/images'

    element_category = ["Logo", "文字", "衬底", "符号元素", "强调突出子部分文字"]

    DatasetCatalog.register("layout_train", lambda: load_coco_json(TRAIN_JSON, image_root=IMAGE_ROOT + '/train', dataset_name="layout_train"))
    MetadataCatalog.get("layout_train").set(thing_classes=element_category,
                                            json_file=TRAIN_JSON,
                                            image_root=IMAGE_ROOT + '/train')

    # Create data loader
    mapper = RADMDatasetMapper(cfg, is_train=True)
    data_loader = build_detection_train_loader(cfg, mapper=mapper)

    # Get one batch
    batch_data = next(iter(data_loader))
    batch = batch_data[0]  # Take first sample

    print("Batch keys:", list(batch.keys()))
    print("Image shape:", batch['image'].shape)
    print("Has text_fea:", 'text_fea' in batch)
    if 'text_fea' in batch:
        print("text_fea keys:", list(batch['text_fea'].keys()))
        print("text_fea feats shape:", batch['text_fea']['feats'].shape)

    # Create RADM model
    from RADM import RADM
    model = RADM(cfg)
    model.eval()

    # Test RADM output
    with torch.no_grad():
        output = model([batch])

    print("\nRADM output type:", type(output))
    print("RADM output length:", len(output) if isinstance(output, list) else "N/A")

    if isinstance(output, list) and output:
        print("First output type:", type(output[0]))
        if isinstance(output[0], dict):
            print("First output keys:", list(output[0].keys()))
            if 'instances' in output[0]:
                instances = output[0]['instances']
                print("Instances type:", type(instances))
                print("Pred boxes shape:", instances.pred_boxes.tensor.shape)
                print("Pred classes shape:", instances.pred_classes.shape)
                print("Pred classes:", instances.pred_classes)


if __name__ == "__main__":
    main()
