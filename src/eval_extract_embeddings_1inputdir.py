import os
import torch
import numpy as np
from tqdm import tqdm
from model_feature_extractor import VisionTransformer
from config import get_eval_extract_embeddings_1inputdir_config
from checkpoint import load_checkpoint
from data_loaders import *
from utils import accuracy, setup_device
from scipy import io
from data_loaders import TBIOMDataLoader_1inputdir

def npy2mat(source_path,output_path,var_name):
    mat = np.load(source_path)
    io.savemat(output_path, {'var_name': mat})

def main():
    print(os.getcwd())
    config = get_eval_extract_embeddings_1inputdir_config()

    # device
    device, device_ids = setup_device(config.n_gpu)


    # create model
    model = VisionTransformer(
             image_size=(config.image_size, config.image_size),
             patch_size=(config.patch_size, config.patch_size),
             emb_dim=config.emb_dim,
             mlp_dim=config.mlp_dim,
             num_heads=config.num_heads,
             num_layers=config.num_layers,
             num_classes=config.num_classes,
             attn_dropout_rate=config.attn_dropout_rate,
             dropout_rate=config.dropout_rate)

    # load checkpoint
    if config.checkpoint_path:
        state_dict = load_checkpoint(config.checkpoint_path)
        model.load_state_dict(state_dict)
        print("Load pretrained weights from {}".format(config.checkpoint_path))

    # send model to device
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # create dataloader
    data_loader = TBIOMDataLoader_1inputdir(
                    data_dir=os.path.join(config.data_dir),
                    image_size=config.image_size,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    split=config.part)
    total_batch = len(data_loader)

    # starting evaluation
    print("Starting evaluation")
    model.eval()

    file_name_list = list(data_loader.dataset.samples)
    # print(file_name_list)
    feature_list = []
    target_list = []
    with torch.no_grad():
        pbar = tqdm(enumerate(data_loader), total=total_batch)
        for batch_idx, (data, target) in pbar:
            pbar.set_description("Batch {:05d}/{:05d}".format(batch_idx, total_batch))

            data = data.to(device)
            target = target.to(device)

            features = model(data)


            for i in range(len(features)):
                feature = features[i].cpu().numpy()
                target_single = target[i].cpu().numpy()
                feature_list.append(feature)
                target_list.append(target_single)



    output_dir = config.output_dir
    os.makedirs(output_dir,exist_ok=True)
    np.save(os.path.join(output_dir,'features.npy'),feature_list)
    with open(os.path.join(output_dir,'filenames.csv'),'a+') as f:
        for file_name in file_name_list:
            f.write(str(file_name[0])+'\n')
            


if __name__ == '__main__':
    main()