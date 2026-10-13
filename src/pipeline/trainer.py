import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import build_segmentation_model, BCEDiceLoss, compute_dice_score, compute_iou_score
from src.data import AponeurosisDataset, discover_training_pairs, make_kfold_splits
from src.utils import get_logger

logger = get_logger("Trainer")


class ModelTrainer:
    """
    Manages 5-fold cross-validation training on Kaggle GPU accelerators.
    Features mixed-precision training, cosine learning rate decay, and checkpointing.
    """
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.cfg.output_dir, exist_ok=True)

    def train_fold(self, fold_idx, train_pairs, val_pairs):
        logger.info(f"--- Starting Fold {fold_idx + 1}/{self.cfg.num_folds} ---")
        logger.info(f"Train samples: {len(train_pairs)}, Val samples: {len(val_pairs)}")

        train_ds = AponeurosisDataset(train_pairs, target_size=self.cfg.image_size, is_train=True)
        val_ds = AponeurosisDataset(val_pairs, target_size=self.cfg.image_size, is_train=False)

        train_loader = DataLoader(
            train_ds, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=True
        )

        model = build_segmentation_model(
            backbone=self.cfg.backbone,
            in_channels=self.cfg.in_channels,
            num_classes=self.cfg.num_classes,
            pretrained=self.cfg.pretrained
        ).to(self.device)

        criterion = BCEDiceLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.epochs, eta_min=self.cfg.min_learning_rate
        )
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        best_val_dice = 0.0
        best_ckpt_path = os.path.join(self.cfg.output_dir, f"model_fold_{fold_idx}.pt")

        for epoch in range(self.cfg.epochs):
            model.train()
            train_loss = 0.0

            for imgs, masks in train_loader:
                imgs = imgs.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits = model(imgs)
                    loss = criterion(logits, masks)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item() * len(imgs)

            scheduler.step()
            train_loss /= len(train_ds)

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_dice = 0.0
            val_iou = 0.0

            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs = imgs.to(self.device, non_blocking=True)
                    masks = masks.to(self.device, non_blocking=True)

                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        logits = model(imgs)
                        loss = criterion(logits, masks)

                    val_loss += loss.item() * len(imgs)
                    probs = torch.sigmoid(logits)
                    val_dice += compute_dice_score(probs, masks) * len(imgs)
                    val_iou += compute_iou_score(probs, masks) * len(imgs)

            val_loss /= len(val_ds)
            val_dice /= len(val_ds)
            val_iou /= len(val_ds)

            logger.info(
                f"Epoch [{epoch+1:02d}/{self.cfg.epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}"
            )

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save({
                    "fold": fold_idx,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                    "config": self.cfg,
                }, best_ckpt_path)

        logger.info(f"Fold {fold_idx + 1} complete. Best Val Dice: {best_val_dice:.4f}")
        return best_val_dice

    def run_cv(self):
        logger.info(f"Scanning training data in {self.cfg.data_dir}...")
        pairs = discover_training_pairs(self.cfg.data_dir)
        logger.info(f"Discovered {len(pairs)} cleaned aponeurosis training pairs.")

        splits = make_kfold_splits(pairs, num_folds=self.cfg.num_folds, seed=self.cfg.seed)
        fold_scores = []

        for f_idx, (train_p, val_p) in enumerate(splits):
            score = self.train_fold(f_idx, train_p, val_p)
            fold_scores.append(score)

        mean_dice = sum(fold_scores) / len(fold_scores)
        logger.info("==========================================")
        logger.info(f"Cross-Validation Complete! Mean Dice: {mean_dice:.4f}")
        logger.info("==========================================")
        return fold_scores
