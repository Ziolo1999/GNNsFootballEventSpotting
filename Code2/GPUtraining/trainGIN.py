import sys
import os
sys.path.append(os.path.abspath('.'))

from data_management.DataManager import CALFData, collateGCN
import torch 
from Model import ContextAwareModel
from helpers.loss import ContextAwareLoss
from modules.train import trainer
import pickle
from dataclasses import dataclass
from helpers.classes import get_K_params

@dataclass
class Args:
    # DATA
    datapath="/project_antwerp/football_games"
    chunk_size = 60
    batch_size = 32
    input_channel = 13
    annotation_nr = 10
    receptive_field = 12
    fps = 5
    K_parameters = get_K_params(chunk_size)
    focused_annotation = None
    generate_augmented_data = True
    class_split = "alive"
    generate_artificial_targets = False
    
    # TRAINING
    chunks_per_epoch = 1824
    lambda_coord=5.0
    lambda_noobj=0.5
    patience=25
    LR=1e-03
    max_epochs=180
    max_num_worker=1
    loglevel='INFO'
    
    # SEGMENTATION MODULE
    feature_multiplier=1
    backbone_player = "GIN"
    load_weights=None
    model_name="Testing_Model"
    dim_capsule=16
    # VLAD pooling if applicable
    vocab_size=None
    pooling=None

    # SPOTTING MODULE
    sgementation_path = None
    freeze_model = None
    spotting_fps = 1

def main():
    args = Args
    collate_fn = collateGCN

    train_dataset = CALFData(split="train", args=args)
    validation_dataset = CALFData(split="validate", args=args)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    validate_loader = torch.utils.data.DataLoader(validation_dataset,
                batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = ContextAwareModel(args=args)
    criterion = ContextAwareLoss(K=train_dataset.K_parameters)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, 
                                betas=(0.9, 0.999), eps=1e-07, 
                                weight_decay=0, amsgrad=False)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=args.patience)

    losses = trainer(train_loader, validate_loader,
                        model, optimizer, scheduler, 
                        criterion,
                        model_name=args.model_name,
                        max_epochs=args.max_epochs, 
                        save_dir=f"/project_antwerp/models/backbone_GIN.pth.tar")

    del train_dataset, validation_dataset, train_loader, validate_loader

    with open(f'/project_antwerp/results/backbone_GIN.pkl', 'wb') as file:
        pickle.dump(losses, file)


if __name__ == "__main__":
    main()