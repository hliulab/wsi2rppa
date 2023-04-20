import argparse
import os
import time
from collections import OrderedDict

import openslide
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models.resnet_custom import resnet50_baseline
from utils.file_utils import save_hdf5
from utils.utils import collate_features

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def compute_w_loader(file_path, output_path, wsi, model,
                     batch_size=8, verbose=0, print_every=20, pretrained=True,
                     custom_downsample=1, target_patch_size=-1):
    """
    args:
        file_path: directory of bag (.h5 file)
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        batch_size: batch_size for computing features in batches
        verbose: level of feedback
        pretrained: use weights pretrained on imagenet
        custom_downsample: custom defined downscale factor of image patches
        target_patch_size: custom defined, rescaled image size before embedding
    """
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained,
                                 custom_downsample=custom_downsample, target_patch_size=target_patch_size,
                                 train_simclr=False, bag_name=None)
    # x, y = dataset[0]
    kwargs = {'num_workers': 32, 'pin_memory': True} if device.type == "cuda" else {}
    loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path, len(loader)))

    extract_time = 0
    to_cpu_time = 0
    load_time = 0
    io_time = 0

    s_time = time.time()
    mode = 'w'
    for count, (batch, coords) in enumerate(loader):
        load_time += time.time() - s_time
        with torch.no_grad():
            if count % print_every == 0:
                print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
            batch = batch.to(device, non_blocking=True)

            s_time = time.time()
            features = model(batch)
            extract_time += time.time() - s_time

            s_time = time.time()
            features = features.cpu().numpy()
            to_cpu_time += time.time() - s_time

            s_time = time.time()
            asset_dict = {'features': features, 'coords': coords}
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            io_time += time.time() - s_time
            mode = 'a'

        s_time = time.time()

    return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default='RESULTS_DIRECTORY')
parser.add_argument('--data_slide_dir', type=str, default="/wsi/tcga/breast_ts")
parser.add_argument('--slide_ext', type=str, default='.svs')
parser.add_argument('--csv_path', type=str, default='RESULTS_DIRECTORY/all_brca_h5.csv')
parser.add_argument('--feat_dir', type=str, default='/wsi/tcga/breast_ts_feats')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--use_simclr', type=bool, default=False)

parser.add_argument('--model_type', type=str)
parser.add_argument('--data_type', type=str, default="tcga_brca")
parser.add_argument('--model_path', type=str)
args = parser.parse_args()


def load_model(model_name, weight_path):

    model = resnet50_baseline()
    if model_name == "ADCO_last":
        check_point = torch.load(weight_path)
        state_dict = check_point["state_dict"]
        new_sd = OrderedDict()
        for k in list(state_dict.keys()):
            # 只要encoder_q
            if k.startswith('encoder_q'):
                new_sd[k[len("encoder_q."):]] = state_dict[k]
        missing_key = model.load_state_dict(new_sd, strict=False)
        assert set(missing_key.unexpected_keys) == {"fc.0.weight", "fc.0.bias", "fc.2.weight", "fc.2.bias"}
        return model

    elif model_name == "MoCo_ft":
        import torchvision.models as models
        model_moco = models.__dict__['resnet50'](num_classes=1)
        model_moco.fc = nn.Sequential()
        check_point = torch.load(weight_path)
        state_dict = check_point["model"]
        msg = model_moco.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == set()
        return model_moco

    elif model_name == "MoCo" or model_name == "MoCov1" or model_name == "Simclr":
        import torchvision.models as models
        model_moco = models.__dict__['resnet50'](num_classes=1)
        model_moco.fc = nn.Sequential()
        check_point = torch.load(weight_path)
        state_dict = check_point["state_dict"]
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('backbone'):
                # remove prefix
                state_dict[k[len("backbone."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = model_moco.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == set()
        return model_moco


if __name__ == '__main__':

    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    bags_dataset = Dataset_All_Bags(csv_path)

    type_path = args.model_type

    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files', type_path), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files', type_path), exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files', type_path))

    model = load_model(args.model_type, args.model_path)

    model = model.to(device)

    model.eval()
    total = len(bags_dataset)

    files = os.listdir(args.data_slide_dir)
    txt_file = None
    for f in files:
        if f.split(".")[-1] == "txt":
            txt_file = f
            break
    assert txt_file is not None
    kidney_info = pd.read_csv(os.path.join(args.data_slide_dir, txt_file), sep="\t")
    slide_to_dir = dict(zip(kidney_info["filename"].values, kidney_info["id"].values))
    for bag_candidate_idx in range(total):
        slide_id = bags_dataset[bag_candidate_idx][:-4]
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', args.data_type, bag_name)
        if not os.path.isfile(h5_file_path):
            continue

        temp_dir = slide_to_dir[slide_id + ".svs"]

        slide_file_path = os.path.join(args.data_slide_dir, temp_dir, slide_id + args.slide_ext)
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)

        if not args.no_auto_skip and slide_id + '.pt' in dest_files:
            print('skipped {}'.format(slide_id))
            continue

        output_path = os.path.join(args.feat_dir, 'h5_files', type_path, bag_name)
        time_start = time.time()
        wsi = openslide.open_slide(slide_file_path)
        output_file_path = compute_w_loader(h5_file_path, output_path, wsi,
                                            model=model, batch_size=args.batch_size, verbose=1, print_every=5,
                                            custom_downsample=args.custom_downsample,
                                            target_patch_size=args.target_patch_size)
        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
        features = torch.randn(1)
        bag_base, _ = os.path.splitext(bag_name)
        torch.save(features, os.path.join(args.feat_dir, 'pt_files', type_path, bag_base + '.pt'))
