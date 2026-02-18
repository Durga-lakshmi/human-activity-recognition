import hydra
import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from datasets import get_dataset,get_dataset_with_minor

#from datasets.hapt.dataset import HAPT
from artifacts.notebooks.EDA.HAPT.hapt_experimental import plot_one_user_raw, plot_one_user_raw_separate_axes, plot_examples_by_activity_12,plot_examples_by_activity_12_new, compute_activity_duration_table,plot_activity_duration_figures,plot_one_user_signal,plot_examples_by_activity_12_signal,plot_one_user_signal_with_label_background
from artifacts.notebooks.EDA.HAPT.split_analysis import run_split_analysis

from tools.analyze_hapt_static_axes import analyze_hapt_static_axes
from tools.signal_viz import plot_features_column_and_save
from models.util import get_model

from trainer import Trainer_HAPT, Trainer_RW
from evaluator import Evaluator_RW
from eval import run_test_eval_HAPT, run_test_eval_RW


from module_training.augmentations import compute_sample_weights
from torch.utils.data import WeightedRandomSampler

from module_training.transition import compute_transition_from_dataset,print_transition_summary, save_transition_matrix


@hydra.main(version_base=None, config_path='config', config_name='defaultp.yaml')
def main(cfg):


    if cfg.name == "default_HAPT":
        # -------------------------------
        #               Device
        # -------------------------------
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # -------------------------------
        #           Dataset info
        # -------------------------------
        print(f"Dataset: {cfg.dataset.name}")
        print(f"[CFG DEBUG] minor_dataloader = {cfg.train.minor_dataloader}")

        # -------------------------------
        #       plot raw data sample
        # -------------------------------
        #plot_one_user_raw(cfg, user_id=1, exp_pick=0, max_len=1000, save_path=cfg.plot_dataset.sample_path)
        #plot_one_user_raw_separate_axes(cfg, user_id=1, exp_pick=0, max_len=None, save_path=cfg.plot_dataset.sample_path)
        #plot_examples_by_activity_12(cfg, user_id=1, exp_pick=0, seconds=4, fs=50, max_examples_per_activity=1, save_dir="./notebooks/EDA/HAPT/by_activity_12")
        #plot_examples_by_activity_12_new(cfg, user_id=1, exp_pick=0, seconds=4, fs=50, max_examples_per_activity=1, save_dir="./notebooks/EDA/HAPT/by_activity_12_new")
        #_ = compute_activity_duration_table(cfg,split="train",fs=50,save_path="./notebooks/EDA/HAPT/by_activity_12_new/activity_duration_train.csv")
        #plot_activity_duration_figures(csv_path="./notebooks/EDA/HAPT/by_activity_12_new/activity_duration_train.csv",save_dir="./notebooks/EDA/HAPT/by_activity_12_new/fig_activity_duration")
        
        #plot_one_user_signal(cfg,user_id=1,exp_pick=0,max_len=4000,fs=50.0,save_path="figs/user01_exp00_phys.png")
        plot_one_user_signal_with_label_background(cfg,user_id=1,exp_pick=0,max_len=19000,fs=50.0,save_path="figs/user01_exp00_bg.png")
        #plot_examples_by_activity_12_signal(cfg,user_id=1,exp_pick=0,seconds=4,fs=50,max_examples_per_activity=1,save_dir="figs/examples_by_act")



        # -------------------------------
        #   Datasets and Dataloaders
        # -------------------------------
        if cfg.train.minor_dataloader:
            train_dataset, train_loader, minor_loader = get_dataset_with_minor(cfg, split="train")
        else:
            train_dataset, train_loader = get_dataset(cfg, split="train")

        # build val/test dataloader with the same normalization stats as train
        val_dataset, val_loader = get_dataset(cfg, split="val")
        test_dataset, test_loader = get_dataset(cfg, split="test")

        ## ----------------------------- Dataset analysis -------------------------------------------------------

        ## 1.Calculate probability transition from train dataset
        #count_raw, count_nogap, P_raw, P_nogap = compute_transition_from_dataset(train_dataset,num_classes=12)
        #CLASS_NAMES = ["WALK","WALK_UP","WALK_DOWN","SIT","STAND","LAY","STAND_TO_SIT","SIT_TO_STAND","SIT_TO_LAY","LAY_TO_SIT","STAND_TO_LAY","LAY_TO_STAND"]
        ## print summary
        #print_transition_summary(P_raw,   CLASS_NAMES, title="P_raw (no cross-gap)")
        #print_transition_summary(P_nogap, CLASS_NAMES, title="P_nogap (gap removed)")

        # save summary
        #out_dir = "./transition_matrixs"
        #save_transition_matrix(P_raw,   out_dir, "P_raw",   CLASS_NAMES)
        #save_transition_matrix(P_nogap, out_dir, "P_nogap", CLASS_NAMES)

        ## 2. Analyzes class imbalance in the training set and constructs weights for both data sampling and the loss function to mitigate its impact during training.
        labels = np.asarray(train_dataset.y, dtype=np.int64)
        # 1) Compute sample_weights from all labels (including -1), as handled in compute_sample_weights
        sample_weights, class_counts, sample_class_weights = compute_sample_weights(labels, power=0.5)
        sampler = WeightedRandomSampler(weights=sample_weights,num_samples=len(sample_weights),replacement=True)
        train_aug_loader = DataLoader(train_dataset,batch_size=cfg.dataset.batch_size,sampler=sampler,shuffle=False)

        # 2) When counting the number of samples per class, ignore -1.
        labels_nonneg = labels[labels >= 0]  
        class_counts = np.bincount(labels_nonneg, minlength=12).astype(np.float32)

        #print("class counts:", class_counts)

        alpha = 0.25  # smoothing factor for loss class weights
        #class_counts = np.bincount(labels, minlength=6).astype(np.float32)
        class_freq = class_counts / class_counts.sum()
        loss_class_weights = (1.0 / (class_freq + 1e-8)) ** alpha
        print("class counts:", class_counts)
        print("sample_class_weights:", sample_class_weights)
        print("class freaq:", class_freq)
        print("loss_class_weights:", loss_class_weights)


        ## 3. Analyze accelerometer axis distribution for static classes in train/test sets.
        #analyze_hapt_static_axes(data_loader=train_loader,static_classes=(3, 4, 5),use_abs=True)
        #analyze_hapt_static_axes(data_loader=val_loader,static_classes=(3, 4, 5),use_abs=True)
        #analyze_hapt_static_axes(data_loader=test_loader,static_classes=(3, 4, 5),use_abs=True)

        ## 4. Diagnose train/val/test splits via label statistics, feature distributions, PCA, and domain separability.
        #X_train_np = train_dataset.X.detach().cpu().numpy()        # (N, T, C)
        #y_train_np = train_dataset.y.detach().cpu().numpy()        # (N,)  state label, include -1
        #X_val_np   = val_dataset.X.detach().cpu().numpy()
        #y_val_np   = val_dataset.y.detach().cpu().numpy()
        #X_test_np  = test_dataset.X.detach().cpu().numpy()
        #y_test_np  = test_dataset.y.detach().cpu().numpy()
        #run_split_analysis(X_train_np, y_train_np,X_val_np, y_val_np,X_test_np,  y_test_np,save_dir=cfg.split_analysis.save_dir)



        
        # --------------------------
        #           Model
        # --------------------------
        model = get_model(cfg)
        model.to(device)

        # --------------------------
        #           Trainer
        # --------------------------

        if cfg.train.minor_dataloader:
            # Purpose: create a dedicated DataLoader for minority classes 
            # to increase their sampling frequency during training
            # enabling explicit oversampling and mitigating class imbalance.
            trainer = Trainer_HAPT(
                cfg=cfg,
                train_loader=train_loader,
                #train_loader=train_aug_loader,
                val_loader=val_loader,
                model=model,
                #evaluator=val_evaluator,
                device=device,
                minor_loader=minor_loader, 
            )
        else:
            trainer = Trainer_HAPT(
                cfg=cfg,
                train_loader=train_loader,
                #train_loader=train_aug_loader,
                val_loader=val_loader,
                model=model,
                #evaluator=val_evaluator,
                device=device,
            )

        # ---------------------------------
        #           Weights & Biases
        # ---------------------------------
        wandb.init(
            #entity=cfg.wandb.init.entity,
            project=cfg.wandb.init.project, 
            name=cfg.wandb.init.name,
            group=cfg.wandb.init.group,
            tags=list(cfg.wandb.init.tags) if "tags" in cfg.wandb.init else None,
            id=cfg.wandb.init.id,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

        wandb.watch(
            model,
            log="gradients",   # 或 "all"
            log_freq=100
        )

        # ---------------------------------
        #           Training
        # ---------------------------------
        trainer.train()

        # ---------------------------------
        #    Final evaluation on test set
        # ---------------------------------
        run_test_eval_HAPT(cfg, model, device)
    else:
        # -------------------------------
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # -------------------------------
        # Dataset info
        # -------------------------------
        print(f"Dataset: {cfg.dataset.name}")

        # -------------------------------
        # Compute train stats
        # -------------------------------
        # mean, std = compute_train_stats(cfg) #run 21 mal -> maybe have problem?


        #plot_one_user_raw(cfg, user_id=1, exp_pick=0, max_len=1000, save_path=cfg.plot_dataset.sample_path)
        # plot_one_user_raw_separate_axes(cfg, user_id=1, exp_pick=0, max_len=18425, save_path=cfg.plot_dataset.sample_path)

        # -------------------------------
        # Datasets and Dataloaders
        # -------------------------------
        train_dataset, train_loader, mean, std = get_dataset(cfg, split="train")
        val_dataset, val_loader = get_dataset(cfg, split="val", mean=mean, std=std)
        test_dataset, test_loader = get_dataset(cfg, split="test", mean=mean, std=std)

        # -------------------------------
        # Inspect one batch
        # -------------------------------
        if cfg.dataset.name == "HAPT":
            for X_batch, y_batch in train_loader:
                print("X batch shape:", X_batch.shape)  # Expected: (batch_size, num_positions, T, 6)
                print("y batch shape:", y_batch.shape)  # Expected: (batch_size,)
        else:
            for X_batch, y_batch in train_loader:
            # X_batch is a dict {position_name: [B, T, C]}
                for pos, data in X_batch.items():
                    print(f"{pos} shape: {data.shape}")  # e.g., [batch_size, window_size, 6]
                
                print("y_batch shape:", y_batch.shape)  # [batch_size]
                break


        # for X_batch, y_batch in train_loader:
        #     print("X batch shape:", X_batch.shape)
        #     print("y batch shape:", y_batch.shape)
        #     break
        
        # --------------------------------------------------
        # Model
        # --------------------------------------------------
        model = get_model(cfg)
        model.to(device)

        # --------------------------------------------------
        # Evaluator (validation)
        # --------------------------------------------------
        val_evaluator = Evaluator_RW(
            cfg=cfg,
            eval_loader=val_loader,
            model=model,
            device=device
        )

        # --------------------------------------------------
        # Weights & Biases
        # --------------------------------------------------
        # wandb.init(
        #     project=cfg.wandb.init.project, 
        #     name=cfg.wandb.init.name,
        #     group=cfg.wandb.init.group,
        #     tags=list(cfg.wandb.init.tags) if "tags" in cfg.wandb.init else None,
        #     id=cfg.wandb.init.id,
        #     config=cfg.wandb.init.config,
        # )
        wandb.init(
            project="human_activity",
            #entity="st192588-university-of-stuttgart",
            # id="mr5tf3ef",              # <-- your run’s ID
            # resume="allow", 
            # config=cfg.wandb.init.config
        )


        # --------------------------------------------------
        # Trainer
        # --------------------------------------------------
        trainer = Trainer_RW(
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            evaluator=val_evaluator,
            device=device,
        )

        trainer.train()
        # trainer.train(resume=True, checkpoint_path = cfg.check_path) 

        # --------------------------------------------------
        # Testing
        # --------------------------------------------------
        
        run_test_eval_RW(cfg, model, device)





if __name__ == "__main__":
    main()

