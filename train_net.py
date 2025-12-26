# ==========================================
# Modified by Shoufa Chen
# ===========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
RADM Training Script.

This script is base on the training script in detectron2/tools.
"""

import os
import itertools
import weakref
from typing import Any, Dict, List, Set
import logging
from collections import OrderedDict

import torch
from fvcore.nn.precise_bn import get_bn_modules

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, create_ddp_model, \
    AMPTrainer, SimpleTrainer, hooks
from detectron2.evaluation import COCOEvaluator, LVISEvaluator, verify_results, DatasetEvaluator, inference_on_dataset, print_csv_format

from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.modeling import build_model
from detectron2.utils.events import EventStorage

from RADM import RADMDatasetMapper, add_radm_config, RADMWithTTA
from RADM.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer
from RADM.layers import PreferenceModel, DDPOTrainer
from RADM.util.box_ops import box_xyxy_to_cxcywh

import pdb


from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json





class Trainer(DefaultTrainer):
    """ Extension of the Trainer class adapted to RADM. """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(DefaultTrainer, self).__init__()  # call grandfather's `__init__` while avoid father's `__init()`
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Set device (similar to RADM detector)
        self.device = torch.device(cfg.MODEL.DEVICE)

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        # DDPO components
        self.use_ddpo = cfg.MODEL.RADM.USE_DDPO
        if self.use_ddpo:
            self.preference_model = PreferenceModel(
                layout_dim=cfg.MODEL.RADM.HIDDEN_DIM,
                text_dim=768,  # RoBERTa embedding dim
                hidden_dim=cfg.MODEL.RADM.DDPO_HIDDEN_DIM
            ).to(self.device)

            # DDPO使用布局坐标作为特征，维度为4 (x1,y1,x2,y2)
            layout_dim = 4  # 布局坐标维度
            self.ddpo_trainer = DDPOTrainer(
                model=model,
                preference_model=PreferenceModel(
                    layout_dim=layout_dim,
                    text_dim=768,  # RoBERTa embedding dim
                    hidden_dim=cfg.MODEL.RADM.DDPO_HIDDEN_DIM
                ).to(self.device),
                beta=cfg.MODEL.RADM.DDPO_BETA,
                sample_size=cfg.MODEL.RADM.DDPO_SAMPLE_SIZE
            )

            # DDPO optimizer
            self.ddpo_optimizer = torch.optim.AdamW(
                self.preference_model.parameters(),
                lr=cfg.MODEL.RADM.DDPO_LR,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY
            )

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        ########## EMA ############
        kwargs = {
            'trainer': weakref.proxy(self),
        }
        kwargs.update(may_get_ema_checkpointer(cfg, model))
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
            # trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results
    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        # setup EMA
        may_build_model_ema(cfg, model)
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if 'lvis' in dataset_name:
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        else:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = RADMDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)
   
    @classmethod
    def build_test_loader(cls, cfg):
        mapper = RADMDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, 'layout_val', mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def ema_test(cls, cfg, model, evaluators=None):
        # model with ema weights
        logger = logging.getLogger("detectron2.trainer")
        if cfg.MODEL_EMA.ENABLED:
            logger.info("Run evaluation with EMA.")
            with apply_model_ema_and_restore(model):
                results = cls.test(cfg, model, evaluators=evaluators)
        else:
            # pdb.set_trace()
            results = cls.test(cfg, model, evaluators=evaluators)
        return results

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = RADMWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        if cfg.MODEL_EMA.ENABLED:
            cls.ema_test(cfg, model, evaluators)
        else:
            res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            EMAHook(self.cfg, self.model) if cfg.MODEL_EMA.ENABLED else None,  # EMA hook
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def train_with_ddpo(self):
        """
        结合扩散模型训练和DDPO优化的训练循环
        """
        logger = logging.getLogger("detectron2")
        logger.info("Starting training with DDPO...")

        # 设置EventStorage上下文 (修复detectron2事件存储错误)
        with EventStorage() as self.storage:
            # 首先进行基础的扩散模型训练
            max_iter = self.max_iter

            # 训练开始欢迎信息
            logger.info("=" * 80)
            logger.info("🎯 RADM + DDPO Training Started!")
            logger.info("🎨 Layout Generation with Reinforcement Learning from Human Preferences")
            logger.info("=" * 80)

            # DDPO训练配置总结
            if self.use_ddpo:
                logger.info("🚀 DDPO Training Configuration:")
                logger.info(f"  📊 Sample Size: {self.cfg.MODEL.RADM.DDPO_SAMPLE_SIZE} (每次DDPO步骤的采样数量)")
                logger.info(f"  🔄 Update Frequency: {self.cfg.MODEL.RADM.DDPO_UPDATE_FREQ} (每N个迭代进行一次DDPO更新)")
                logger.info(f"  β Beta: {self.cfg.MODEL.RADM.DDPO_BETA} (DPO损失的正则化参数)")
                logger.info(f"  🧠 Hidden Dim: {self.cfg.MODEL.RADM.DDPO_HIDDEN_DIM} (偏好模型隐藏层维度)")
                logger.info(f"  📈 Total Expected DDPO Steps: {max_iter // self.cfg.MODEL.RADM.DDPO_UPDATE_FREQ}")
                logger.info("=" * 70)
            # diffusion_warmup_iters = int(max_iter * 0.3)  # 前30%用于纯扩散训练
            diffusion_warmup_iters = 0 # 直接测试rl train

            # 第一阶段：纯扩散模型训练
            if diffusion_warmup_iters > 0:
                logger.info("🌡️ Phase 1: Diffusion-only Warmup Training")
                logger.info(f"   Training diffusion model for {diffusion_warmup_iters} iterations to stabilize...")
                for self.iter in range(diffusion_warmup_iters):
                    with EventStorage() as storage:
                        self.storage = storage
                        self.run_step()
                    if self.iter % 1000 == 0 and self.iter > 0:
                        logger.info(f"   🌡️ Diffusion warmup: iteration {self.iter}/{diffusion_warmup_iters}")
                logger.info("✅ Phase 1 completed: Diffusion model warmed up!")
            else:
                logger.info("⚡ Skipping diffusion warmup (set to 0 iterations)")

            # 第二阶段：扩散模型 + DDPO 联合训练
            joint_iters = max_iter - diffusion_warmup_iters
            logger.info("🤖 Phase 2: Joint Diffusion + DDPO Training")
            logger.info(f"   Training both diffusion model and DDPO preference model for {joint_iters} iterations")
        ddpo_step_count = 0
        for self.iter in range(diffusion_warmup_iters, max_iter):
            # 扩散模型训练步骤
            with EventStorage() as storage:
                self.storage = storage
                self.run_step()

            # DDPO训练步骤 (每隔一定迭代进行一次)
            if self.iter % self.cfg.MODEL.RADM.DDPO_UPDATE_FREQ == 0:
                ddpo_loss = self.ddpo_step()
                ddpo_step_count += 1

                        # 记录详细的DDPO训练日志
                if ddpo_step_count % 5 == 0:  # 每5个DDPO步骤记录一次
                    progress_pct = (self.iter / max_iter) * 100
                    logger.info("3d"
                                "2d")

            # 每1000次迭代记录总体进度
            if self.iter % 1000 == 0 and self.iter > 0:
                elapsed_iters = self.iter - diffusion_warmup_iters
                ddpo_freq = self.cfg.MODEL.RADM.DDPO_UPDATE_FREQ
                expected_ddpo_steps = elapsed_iters // ddpo_freq
                progress_pct = (self.iter / max_iter) * 100

                logger.info("2d"
                            "4d"
                            "4d"
                            "3.1f")

        # 训练完成总结
        logger.info("=" * 80)
        logger.info("🎉 DDPO Training Completed Successfully!")
        logger.info("=" * 80)
        logger.info("📊 Training Summary:")
        logger.info(f"  🔢 Total iterations: {max_iter:,}")
        logger.info(f"  🌡️ Diffusion warmup: {diffusion_warmup_iters:,} iterations")
        logger.info(f"  🤖 DDPO training: {max_iter - diffusion_warmup_iters:,} iterations")
        logger.info(f"  📈 Total DDPO steps: {ddpo_step_count:,}")
        logger.info(f"  🔄 DDPO update frequency: every {self.cfg.MODEL.RADM.DDPO_UPDATE_FREQ} iterations")
        logger.info(f"  🎯 Final DDPO samples per step: {self.cfg.MODEL.RADM.DDPO_SAMPLE_SIZE}")
        logger.info("=" * 80)
        logger.info("💾 Model saved to: ./output_rl/")
        logger.info("📈 Training logs available in the output directory")
        logger.info("=" * 80)

    def ddpo_step(self):
        """
        单步DDPO训练
        """
        import logging
        logger = logging.getLogger("detectron2")
        self.preference_model.train()

        # 从训练数据中采样一个batch
        try:
            batch = next(iter(self.data_loader))
        except StopIteration:
            # 如果数据加载器耗尽，重新开始
            self.data_loader_iter = iter(self.data_loader)
            batch = next(iter(self.data_loader))

        # DDPO在一个batch元素上操作，使用第一个元素
        batch_item = batch[0]  # 单个batch元素 (dict)
        batch_list = [batch_item]  # 包装成list以符合detectron2格式

        # 预处理数据
        images, images_whwh = self.model.preprocess_image(batch_list)

        # 获取文本特征 (只处理第一个batch元素)
        text_features = batch_item["text_fea"]['feats'].to(self.device).unsqueeze(0)  # [1, text_dim]
        txt_mask = batch_item["text_mask"].to(self.device).unsqueeze(0)  # [1, seq_len]

        # 获取ground truth奖励 (只处理第一个batch元素)
        gt_instances = [batch_item["instances"].to(self.device)]
        gt_rewards = self.compute_gt_rewards(gt_instances, images_whwh)

        # 记录详细的奖励统计信息
        reward_mean = gt_rewards.mean().item()
        reward_std = gt_rewards.std().item()
        reward_min = gt_rewards.min().item()
        reward_max = gt_rewards.max().item()
        logger.debug(f"🎯 DDPO Reward Statistics: "
                        f"Mean={reward_mean:.4f}, Std={reward_std:.4f}, "
                        f"Min={reward_min:.4f}, Max={reward_max:.4f}"
                    )

        # 使用真正的DDPO：采样时记录完整轨迹用于replay
        logger.debug(f"🔄 DDPO sampling {self.ddpo_trainer.sample_size} layouts with trajectory recording...")
        samples_with_trajectory = self.ddpo_trainer.sample_layouts_with_trajectory(
            batch_list, text_features.squeeze(0), txt_mask.squeeze(0), num_samples=self.ddpo_trainer.sample_size
        )
        logger.debug(f"✅ DDPO sampling completed: {len(samples_with_trajectory)} samples with full trajectories")

        # 计算DDPO损失 (使用replay模式比较同一个y在不同策略下的概率)
        # 注意：传递正确的维度 - text_features [batch_size, text_dim], txt_mask [batch_size, seq_len]
        logger.debug("🧮 Computing DDPO loss with preference model (replay mode)...")
        ddpo_loss = self.ddpo_trainer.compute_ddpo_loss(
            samples_with_trajectory, batch, text_features, txt_mask, gt_rewards,
            chosen_indices=None, rejected_indices=None  # 暂时使用单样本模式
        )
        logger.debug(f"ddpo_loss.item() : {ddpo_loss}")

        # 反向传播和优化
        self.ddpo_optimizer.zero_grad()
        ddpo_loss.backward()
        self.ddpo_optimizer.step()

    def compute_gt_rewards(self, gt_instances, images_whwh):
        """
        计算ground truth的奖励分数 (基于布局质量指标)
        """
        rewards = []
        for i, instances in enumerate(gt_instances):
            # instances是Instances对象，gt_boxes是tensor
            if len(instances) == 0:
                reward = 0.0
            else:
                # instances.gt_boxes 是 Boxes 对象，需要 .tensor 获取底层tensor
                boxes_tensor = instances.gt_boxes.tensor  # [num_boxes, 4] (x1, y1, x2, y2)

                # 计算每个bbox的面积
                widths = boxes_tensor[:, 2] - boxes_tensor[:, 0]  # x2 - x1
                heights = boxes_tensor[:, 3] - boxes_tensor[:, 1]  # y2 - y1
                areas = widths * heights

                # 计算平均面积 (归一化到图像尺寸)
                img_area = images_whwh[i][0] * images_whwh[i][1]  # 图像面积 (w*h)
                avg_area_ratio = (areas.mean().item() / img_area) if len(areas) > 0 else 0.0

                # 奖励计算：基于bbox数量和相对面积
                num_boxes = len(boxes_tensor)

                # 数量奖励：鼓励适中的bbox数量 (5个左右最佳)
                if num_boxes <= 5:
                    count_reward = num_boxes * 0.2  # 0-1.0
                else:
                    count_reward = max(0, 2.0 - num_boxes * 0.1)  # 惩罚过多bbox

                # 面积奖励：鼓励适中的bbox大小 (0.05-0.3的相对面积最佳)
                if 0.05 <= avg_area_ratio <= 0.3:
                    area_reward = 1.0
                elif avg_area_ratio < 0.05:
                    area_reward = avg_area_ratio / 0.05  # 太小惩罚
                else:
                    area_reward = max(0, 1.0 - (avg_area_ratio - 0.3) / 0.2)  # 太大惩罚

                reward = (count_reward + area_reward) / 2.0  # 平均分数 0-1

            rewards.append(reward)

        return torch.tensor(rewards, device=self.device).unsqueeze(0).repeat(self.ddpo_trainer.sample_size, 1)

    def results_to_layout_features(self, results, images_whwh):
        """
        将推理结果转换为布局特征表示
        Args:
            results: 推理结果列表，每个元素包含预测的boxes, scores, classes
            images_whwh: 图像尺寸 [batch_size, 4] (w, h, w, h)
        Returns:
            layout_features: [batch_size, num_proposals, 4] 布局框坐标 (cx, cy, w, h)
        """
        batch_size = len(results)
        num_proposals = self.model.num_proposals
        layout_features = torch.zeros(batch_size, num_proposals, 4, device=self.device)

        for i, result in enumerate(results):
            if isinstance(result, dict) and "instances" in result:
                instances = result["instances"]
            else:
                instances = result

            if len(instances) > 0:
                # 获取预测框 (x1, y1, x2, y2)
                boxes = instances.pred_boxes.tensor  # [num_instances, 4]

                # 转换为中心坐标格式 (cx, cy, w, h)
                boxes_center = box_xyxy_to_cxcywh(boxes)  # [num_instances, 4]

                # 归一化到0-1范围
                img_whwh = images_whwh[i]  # [4] (w, h, w, h)
                boxes_center = boxes_center / img_whwh[:4]

                # 填充到固定大小
                num_instances = min(len(boxes_center), num_proposals)
                layout_features[i, :num_instances] = boxes_center[:num_instances]

                # 剩余位置用零填充 (已经在初始化时完成)

        return layout_features
    
# add layout register
def register_layout(cfg):
    DATASET_ROOT = cfg.DATASETS.DATASET_PATH
    ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
    TRAIN_JSON = os.path.join(ANN_ROOT, 'train.json')
    VAL_JSON = os.path.join(ANN_ROOT, 'test.json')

    IMAGE_ROOT = os.path.join(DATASET_ROOT, 'images')
    TRAIN_PATH = os.path.join(IMAGE_ROOT, 'train')
    VAL_PATH = os.path.join(IMAGE_ROOT, 'test')


    element_category = ["Logo", "文字", "衬底", "符号元素", "强调突出子部分文字"]

    DatasetCatalog.register("layout_train", lambda: load_coco_json(TRAIN_JSON, image_root=TRAIN_PATH, dataset_name="layout_train"))
    MetadataCatalog.get("layout_train").set(thing_classes = element_category,
                                                        json_file=TRAIN_JSON,
                                                        image_root=TRAIN_PATH)
    
    DatasetCatalog.register("layout_val", lambda: load_coco_json(VAL_JSON, image_root=VAL_PATH, dataset_name="layout_val"))
    MetadataCatalog.get("layout_val").set(thing_classes=element_category,
                                                    json_file=VAL_JSON,
                                                    image_root=VAL_PATH)
# add done

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_radm_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    register_layout(cfg)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        kwargs = may_get_ema_checkpointer(cfg, model)
        if cfg.MODEL_EMA.ENABLED:
            EMADetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(cfg.MODEL.WEIGHTS,
                                                                                              resume=args.resume)
        else:
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(cfg.MODEL.WEIGHTS,
                                                                                           resume=args.resume)
        res = Trainer.ema_test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    if cfg.MODEL.RADM.USE_DDPO:
        return trainer.train_with_ddpo()
    else:
        return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
