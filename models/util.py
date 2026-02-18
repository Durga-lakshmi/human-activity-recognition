from .hapt_lstm.model import HAPTLSTM
from .deepconvlstm.model import DeepConvLSTM,DeepConvBiLSTMAtt,TwoHeadDeepConvLSTM
from .lstm_conv.model import LSTMConv
from .twohead.model import TwoHeadHAR
from .tcn.model import TCN
from .cnn.model import TemporalCNN
from .cnntransformer.model import CNNLightTransformer
from .cnn_tcn.model import CNN_TCN_HAR
from .har_lstm.model import HARLSTM
from .singlehar.model import SingleHAR

from omegaconf import ListConfig
from collections import OrderedDict
import torch

def get_model(cfg):
    if cfg.model.name == 'HAPTLSTM':
        model = HAPTLSTM(
            input_dim=cfg.model.input_dim,
            num_classes=cfg.num_classes,
            window_size=cfg.dataset.window_size
        )
    elif cfg.model.name == 'HARLSTM':
        model = HARLSTM(
            num_channels=cfg.model.num_channels,
            num_positions=cfg.model.num_positions,
            positions=cfg.model.positions,
            num_classes=cfg.model.num_classes,
            hidden_dim=cfg.model.hidden_dim,
            dropout=cfg.model.dropout
        )

    elif cfg.model.name == 'DEEPCONVLSTM':
        return DeepConvLSTM(
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        conv_channels=cfg.model.conv_channels,
        lstm_hidden=cfg.model.lstm.hidden_size,
        lstm_layers=cfg.model.lstm.num_layers,
        dropout=cfg.model.dropout,
    )

    elif cfg.model.name == 'DEEPCONVBILSTMATT':
        return DeepConvBiLSTMAtt(
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        conv_channels=cfg.model.conv_channels,
        lstm_hidden=cfg.model.lstm_hidden,
        lstm_layers=cfg.model.lstm_layers,
        dropout=cfg.model.dropout,
        bidirectional=True,
        use_attention=True,
    )

    elif cfg.model.name == 'DEEPCONV2HEAD':
        return TwoHeadDeepConvLSTM(
        in_channels=6,                     # HAPT: acc(3) + gyro(3)
        num_states=6,                      # WALKING, UP, DOWN, SITTING, STANDING, LAYING
        conv_channels=64,   # e.g. 64
        lstm_hidden=128,       # e.g. 128
        lstm_layers=2,       # e.g. 2
        dropout=0,               # e.g. 0.5
        bidirectional=getattr(cfg.model, "bidirectional", False),
    )


    elif cfg.model.name == 'LSTMCONV':
        return LSTMConv(
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,      # 你的类别数
        lstm_hidden=cfg.model.lstm_hidden,
        lstm_layers=cfg.model.lstm_layers,
        conv_channels=cfg.model.conv_channels,
        kernel_size=cfg.model.kernel_size,
        lstm_dropout=cfg.model.lstm_dropout,
        head_dropout=cfg.model.head_dropout,
        bidirectional=cfg.model.bidirectional, # 6轴常用更稳
        pool=cfg.model.pool,         # 默认更稳
    )

    elif cfg.model.name == 'TWOHEADHAR':
        # 你现在的 pipeline 若只支持返回一个 logits（num_classes），
        # 先把 num_fine_classes 设为 cfg.num_classes
        # coarse head（2类）会在 Stage2 用到，但不影响初始化
        return TwoHeadHAR(
            in_ch=getattr(cfg.model, "in_ch", getattr(cfg.model, "input_dim", 6)),
            num_fine=getattr(cfg.model, "num_fine_classes", cfg.num_classes),
            dropout=getattr(cfg.model, "dropout", 0.2),
        )


    elif cfg.model.name == 'TCN':
        return TCN(
            in_channels=cfg.model.in_channels,
            num_classes=cfg.model.num_classes,
            hidden_channels=cfg.model.hidden_channels,
            num_layers=cfg.model.num_layers,
            kernel_size=cfg.model.kernel_size,
            dropout=cfg.model.dropout,
        )

    if cfg.model.name == "CNN":
        model = TemporalCNN(
            in_channels=cfg.model.in_channels,
            num_classes=cfg.model.num_classes,
            dropout=cfg.model.dropout,
        )
        return model

    if cfg.model.name == "CNN_LIGHT_TRANSFORMER":
        model = CNNLightTransformer(
            in_channels=cfg.model.in_channels,
            num_classes=cfg.model.num_classes,
            d_model=cfg.model.d_model,
            nhead=cfg.model.nhead,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
        )
        return model

      # ===== 新增：CNN + TCN 组合模型 =====
    elif cfg.model.name == "CNN_TCN":
        # 直接把 tcn_dilations 当 list 用
        raw_dils = getattr(cfg.model, "tcn_dilations", [1, 2])

        if isinstance(raw_dils, (list, ListConfig)):
            dilations = [int(x) for x in raw_dils]
        else:
            # 兼容万一哪里传成字符串的情况
            s = str(raw_dils).strip().strip("[]")
            dilations = [int(x) for x in s.split(",") if x.strip()]

        model = CNN_TCN_HAR(
            in_channels=cfg.model.in_channels,
            num_classes=cfg.model.num_classes,
            cnn_channels=getattr(cfg.model, "cnn_channels", 64),
            tcn_channels=getattr(cfg.model, "tcn_channels", 128),
            dropout=cfg.model.tcn_dropout,
            cnn_kernel_size=getattr(cfg.model, "cnn_kernel_size", 5),
            tcn_kernel_size=getattr(cfg.model, "tcn_kernel_size", 3),
            tcn_dilations=dilations,
            pooling=getattr(cfg.model, "pooling", "mean"),
            # ===== 新增：从 cfg 里读取位姿 / 物理 head 开关 =====
            use_pose=getattr(cfg.model, "use_pose", False),
            pose_kernel_size=getattr(cfg.model, "pose_kernel_size", 25),
            pose_detach_gravity=getattr(cfg.model, "pose_detach_gravity", True),
            use_phys_head=getattr(cfg.model, "use_phys_head", False),
            use_posture=True,       # 开启坐/站姿态特征
        )

        print("cfg.model.in_channels =", cfg.model.in_channels)
        print("window_size =", cfg.dataset.sliding_window.window_size)
            
        print_model_summary(model,input_size=(1, cfg.dataset.sliding_window.window_size,cfg.model.in_channels))
        
        return model

    elif cfg.model.name == 'HARLSTM':
        model = HARLSTM(
            num_channels=cfg.model.num_channels,
            num_positions=cfg.model.num_positions,
            positions=cfg.model.positions,
            num_classes=cfg.model.num_classes,
            hidden_dim=cfg.model.hidden_dim,
            dropout=cfg.model.dropout
        )
        print(f'Model "{cfg.model.name}" summary:')
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                total_params += num_params
                print(f"{name:40s} | {num_params:,}")    
        print(f"\nTotal Trainable Parameters: {total_params:,}")
        return model

    elif cfg.model.name == 'SingleHAR':
        model = SingleHAR()
        print(f'Model "{cfg.model.name}" summary:')
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                total_params += num_params
                print(f"{name:40s} | {num_params:,}")    
        print(f"\nTotal Trainable Parameters: {total_params:,}")
        return model

    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")



def print_model_summary(
    model: torch.nn.Module,
    input_size=None,          # e.g. (1, 6, 128)
    print_layer_params=True,
    max_depth=None,
):
    """
    通用模型结构 + 参数量打印工具
    - input_size: 仅用于 forward 检查（可选）
    - max_depth: 限制打印层级，None 表示全部
    """

    total_params = 0
    trainable_params = 0

    print("\n================ Model Summary ================\n")
    print(model)
    print("\n---------------- Parameter Details ----------------")

    for name, param in model.named_parameters():
        num = param.numel()
        total_params += num
        if param.requires_grad:
            trainable_params += num

        if print_layer_params:
            if max_depth is not None:
                depth = name.count(".")
                if depth >= max_depth:
                    continue

            print(
                f"{name:<60} "
                f"shape={tuple(param.shape)!s:<20} "
                f"params={num:>10,d} "
                f"{'[trainable]' if param.requires_grad else '[frozen]'}"
            )

    print("\n---------------- Summary ----------------")
    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")
    print(f"Non-trainable params : {total_params - trainable_params:,}")

    if input_size is not None:
        print("\n---------------- Forward Check ----------------")
        try:
            x = torch.zeros(*input_size)
            with torch.no_grad():
                y = model(x)

            # 1) 如果是 dict：逐个 key 打印 tensor 形状
            if isinstance(y, dict):
                for k, v in y.items():
                    if torch.is_tensor(v):
                        print(f"{k}: shape={tuple(v.shape)}")
                    else:
                        print(f"{k}: type={type(v)} (non-tensor)")
            # 2) 如果是 tuple / list：沿用你原来的逻辑
            elif isinstance(y, (tuple, list)):
                for i, yi in enumerate(y):
                    if torch.is_tensor(yi):
                        print(f"Output[{i}] shape: {tuple(yi.shape)}")
                    else:
                        print(f"Output[{i}] type: {type(yi)} (non-tensor)")
            # 3) 其余情况：比如直接是 tensor
            elif torch.is_tensor(y):
                print(f"Output shape: {tuple(y.shape)}")
            else:
                print(f"Output type: {type(y)} (non-tensor, no shape)")
        except Exception as e:
            print("Forward check failed:", e)

    print("\n================================================\n")
