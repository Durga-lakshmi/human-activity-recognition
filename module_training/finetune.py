import torch

# ====================================================
# utility function to freeze the backbone
# ====================================================
def freeze_backbone(model):
    # Freeze every module inside the backbone
    if hasattr(model, "backbone"):
        for m in model.backbone:          # m refers to submodules such as pose / cnn / tcn / posture
            for p in m.parameters():
                p.requires_grad = False

    # Ensure the classifier remains trainable
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True

    if hasattr(model, "head_trans"):
        for p in model.head_trans.parameters():
            p.requires_grad = True

    if hasattr(model, "head_34"):
        for p in model.head_34.parameters():
            p.requires_grad = True

    if hasattr(model, "head_phys"):
        for p in model.head_phys.parameters():
            p.requires_grad = True



# ====================================================
#  optimizer that only updates the "heads"
#   (learning rate reduced to 1/5 of the original)
# ====================================================
def build_finetune_optimizer(model, cfg):
    """
    Fine-tune only the classifier and auxiliary heads.
    Assumes the original learning rate is cfg.train.lr
    and weight_decay is cfg.train.weight_decay.
    """
    base_lr = cfg.train.lr
    wd = getattr(cfg.train, "weight_decay", 0.0)

    param_groups = []

    # Main classification head
    if hasattr(model, "classifier"):
        param_groups.append({"params": model.classifier.parameters()})

    # Transition auxiliary head
    if hasattr(model, "head_trans"):
        param_groups.append({"params": model.head_trans.parameters()})

    # Auxiliary binary head for classes 3/4
    if hasattr(model, "head_34"):
        param_groups.append({"params": model.head_34.parameters()})

    # Physical head: comment out this block if you do not want to fine-tune it
    if hasattr(model, "head_phys"):
        param_groups.append({"params": model.head_phys.parameters()})

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=base_lr * 0.2,        # learning rate = 1/5 of the original
        weight_decay=wd,
    )
    return optimizer
