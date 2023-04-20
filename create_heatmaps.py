from __future__ import print_function

import argparse
import pdb
import os
import pandas as pd
from utils.utils import *
import h5py
import yaml
from wsi_core.batch_process_utils import initialize_df
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches
from utils.file_utils import save_hdf5
from models.att_model import CLAM_SB

parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code')
parser.add_argument('--overlap', type=float, default=None)
parser.add_argument('--config_file', type=str, default="config_template_new.yaml")
args = parser.parse_args()


def infer_single_slide(model, features):
    features = features.to(device)
    with torch.no_grad():
        if isinstance(model, CLAM_SB):
            A = model(features, attention_only=True)
            A = A.view(-1, 1).cpu().numpy()

        else:
            raise NotImplementedError

    return A


def load_params(df_entry, params):
    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key]
            val = dtype(val)
            if isinstance(val, str):
                if len(val) > 0:
                    params[key] = val
            elif not np.isnan(val):
                params[key] = val
            else:
                pdb.set_trace()

    return params


def parse_config_dict(args, config_dict):
    if args.save_exp_code is not None:
        config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
    if args.overlap is not None:
        config_dict['patching_arguments']['overlap'] = args.overlap
    return config_dict


def load_model(model_name, weight_path):
    if model_name == "MoCo":
        import torchvision.models as models
        model_moco = models.__dict__['resnet50'](num_classes=1)
        model_moco.fc = nn.Sequential()
        check_point = torch.load(weight_path, map_location='cuda:1')
        state_dict = check_point["model"]
        msg = model_moco.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == set()
        return model_moco


if __name__ == '__main__':
    config_path = os.path.join('heatmaps/configs', args.config_file)
    config_dict = yaml.safe_load(open(config_path, 'r'))
    config_dict = parse_config_dict(args, config_dict)

    for key, value in config_dict.items():
        if isinstance(value, dict):
            print('\n' + key)
            for value_key, value_value in value.items():
                print(value_key + " : " + str(value_value))
        else:
            print('\n' + key + " : " + str(value))

    args = config_dict
    patch_args = argparse.Namespace(**args['patching_arguments'])
    data_args = argparse.Namespace(**args['data_arguments'])
    model_args = args['model_arguments']
    model_args.update({'n_classes': args['exp_arguments']['n_classes']})
    model_args = argparse.Namespace(**model_args)
    exp_args = argparse.Namespace(**args['exp_arguments'])
    heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
    sample_args = argparse.Namespace(**args['sample_arguments'])

    patch_size = tuple([patch_args.patch_size for i in range(2)])
    step_size = tuple((np.array(patch_size) * (1 - patch_args.overlap)).astype(int))
    print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1],
                                                                                  patch_args.overlap, step_size[0],
                                                                                  step_size[1]))

    preset = data_args.preset
    def_seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': True,
                      'keep_ids': 'none', 'exclude_ids': 'none'}
    def_filter_params = {'a_t': 100.0, 'a_h': 3.0, 'max_n_holes': 2048}
    def_vis_params = {'vis_level': -1, 'line_thickness': 250}
    def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if preset is not None:
        preset_df = pd.read_csv(preset)
        for key in def_seg_params.keys():
            def_seg_params[key] = preset_df.loc[0, key]

        for key in def_filter_params.keys():
            def_filter_params[key] = preset_df.loc[0, key]

        for key in def_vis_params.keys():
            def_vis_params[key] = preset_df.loc[0, key]

        for key in def_patch_params.keys():
            def_patch_params[key] = preset_df.loc[0, key]

    slide = ''  # slide name
    slide_id = slide.split('.svs')[0]
    slides = [slide_id + '.svs']
    df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params,
                       use_heatmap_args=False)

    mask = df['process'] == 1
    process_stack = df[mask].reset_index(drop=True)
    total = len(process_stack)
    print('\nlist of slides to process: ')
    print(process_stack.head(len(process_stack)))

    print('\ninitializing model from checkpoint')
    ckpt_path = model_args.ckpt_path
    print('\nckpt path: {}'.format(ckpt_path))

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    print('Init Model')
    model = CLAM_SB(n_classes=2)
    cp = torch.load(args.weight_path, map_location=device)
    for k in list(cp['model'].keys()):
        if '3' in k:
            cp['model'][k.replace('3', '2')] = cp['model'][k]
            del cp['model'][k]
        if k == 'classifiers.weight':
            cp['model']['classifiers.0.weight'] = cp['model'][k]
            del cp['model'][k]
        if k == 'classifiers.bias':
            cp['model']['classifiers.0.bias'] = cp['model'][k]
            del cp['model'][k]

    model.load_state_dict(cp['model'])
    model = model.to(device)
    feature_extractor = load_model(model_name=args.model_name,
                                   weight_path=args.weight_path).to(device)
    feature_extractor.eval()
    print('Done!')
    os.makedirs(exp_args.production_save_dir, exist_ok=True)
    os.makedirs(exp_args.raw_save_dir, exist_ok=True)
    blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size,
                         'custom_downsample': patch_args.custom_downsample, 'level': patch_args.patch_level,
                         'use_center_shift': heatmap_args.use_center_shift}

    for i in range(len(process_stack)):
        slide_name = process_stack.loc[i, 'slide_id']
        if data_args.slide_ext not in slide_name and 'tiff' not in slide_name:
            slide_name += data_args.slide_ext
        print('\nprocessing: ', slide_name)

        slide_id = slide_name.replace('.svs', '')

        p_slide_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, slide_id + '123')
        os.makedirs(p_slide_save_dir, exist_ok=True)

        r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, slide_id + '123')
        os.makedirs(r_slide_save_dir, exist_ok=True)

        if heatmap_args.use_roi:
            x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i, 'x2']
            y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
            top_left = (int(x1), int(y1))
            bot_right = (int(x2), int(y2))
        else:
            top_left = None
            bot_right = None

        print('slide id: ', slide_id)
        print('top left: ', top_left, ' bot right: ', bot_right)
        if isinstance(data_args.data_dir, str):
            slide_path = os.path.join(data_args.data_dir, slide_name)
        elif isinstance(data_args.data_dir, dict):
            data_dir_key = process_stack.loc[i, data_args.data_dir_key]
            slide_path = os.path.join(data_args.data_dir[data_dir_key], slide_name)
        else:
            raise NotImplementedError

        mask_file = os.path.join(r_slide_save_dir, slide_id + '_mask.pkl')

        # Load segmentation and filter parameters
        seg_params = def_seg_params.copy()
        filter_params = def_filter_params.copy()
        vis_params = def_vis_params.copy()

        seg_params = load_params(process_stack.loc[i], seg_params)
        filter_params = load_params(process_stack.loc[i], filter_params)
        vis_params = load_params(process_stack.loc[i], vis_params)

        keep_ids = str(seg_params['keep_ids'])
        if len(keep_ids) > 0 and keep_ids != 'none':
            seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
        else:
            seg_params['keep_ids'] = []

        exclude_ids = str(seg_params['exclude_ids'])
        if len(exclude_ids) > 0 and exclude_ids != 'none':
            seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
        else:
            seg_params['exclude_ids'] = []

        for key, val in seg_params.items():
            print('{}: {}'.format(key, val))

        for key, val in filter_params.items():
            print('{}: {}'.format(key, val))

        for key, val in vis_params.items():
            print('{}: {}'.format(key, val))

        print('Initializing WSI object')
        wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params,
                                    filter_params=filter_params)
        print('Done!')

        wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]

        # the actual patch size for heatmap visualization should be the patch size * downsample factor * custom
        # down sample factor
        vis_patch_size = tuple(
            (np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))
        print(f"vis_patch_size：{vis_patch_size}")

        block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
        mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
        if vis_params['vis_level'] < 0:
            best_level = wsi_object.wsi.get_best_level_for_downsample(32)
            vis_params['vis_level'] = best_level
        mask = wsi_object.visWSI(**vis_params, number_contours=True)
        mask.save(mask_path)
        features_path = os.path.join(r_slide_save_dir, slide_id + '.pt')
        h5_path = os.path.join(r_slide_save_dir, slide_id + '.h5')
        exp_args.batch_size = 64
        ##### check if h5_features_file exists ######
        if not os.path.isfile(h5_path):
            _, _, wsi_object = compute_from_patches(wsi_object=wsi_object,
                                                    model=model,
                                                    feature_extractor=feature_extractor,
                                                    batch_size=exp_args.batch_size, **blocky_wsi_kwargs,
                                                    attn_save_path=None, feat_save_path=h5_path,
                                                    ref_scores=None, device=device)

        ##### check if pt_features_file exists ######
        if not os.path.isfile(features_path):
            file = h5py.File(h5_path, "r")
            features = torch.tensor(file['features'][:])
            torch.save(features, features_path)
            file.close()

        # load features
        features = torch.load(features_path, map_location=device)
        process_stack.loc[i, 'bag_size'] = len(features)

        wsi_object.saveSegmentation(mask_file)
        A = infer_single_slide(model, features)
        del features

        if not os.path.isfile(block_map_save_path):
            file = h5py.File(h5_path, "r")
            coords = file['coords'][:]
            file.close()
            asset_dict = {'attention_scores': A, 'coords': coords}
            block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')

        os.makedirs('heatmaps/results/', exist_ok=True)
        if data_args.process_list is not None:
            process_stack.to_csv('heatmaps/results/{}.csv'.format(data_args.process_list.replace('.csv', '')),
                                 index=False)
        else:
            process_stack.to_csv('heatmaps/results/{}.csv'.format(exp_args.save_exp_code), index=False)

        file = h5py.File(block_map_save_path, 'r')
        dset = file['attention_scores']
        coord_dset = file['coords']
        scores = dset[:]
        coords = coord_dset[:]
        file.close()

        samples = sample_args.samples

        wsi_kwargs = {'top_left': top_left, 'bot_right': bot_right, 'patch_size': patch_size, 'step_size': step_size,
                      'custom_downsample': patch_args.custom_downsample, 'level': patch_args.patch_level,
                      'use_center_shift': heatmap_args.use_center_shift}

        heatmap_save_name = '{}_blockmap.tiff'.format(slide_id)
        if os.path.isfile(os.path.join(r_slide_save_dir, heatmap_save_name)):
            pass
        else:
            heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap,
                                  alpha=heatmap_args.alpha, use_holes=True, binarize=False, vis_level=-1,
                                  blank_canvas=False,
                                  thresh=-1, patch_size=vis_patch_size, convert_to_percentiles=True)

            heatmap.save(os.path.join(r_slide_save_dir, '{}_blockmap.png'.format(slide_id)))
            del heatmap

        save_path = os.path.join(r_slide_save_dir,
                                 '{}_{}_roi_{}.h5'.format(slide_id, patch_args.overlap, heatmap_args.use_roi))

        if heatmap_args.use_ref_scores:
            ref_scores = scores
        else:
            ref_scores = None

        if heatmap_args.calc_heatmap:
            compute_from_patches(wsi_object=wsi_object, model=model, device=device,
                                 feature_extractor=feature_extractor, batch_size=exp_args.batch_size, **wsi_kwargs,
                                 attn_save_path=save_path, ref_scores=ref_scores)

        if not os.path.isfile(save_path):
            print('heatmap {} not found'.format(save_path))
            if heatmap_args.use_roi:
                save_path_full = os.path.join(r_slide_save_dir,
                                              '{}_{}_roi_False.h5'.format(slide_id, patch_args.overlap))
                print('found heatmap for whole slide')
                save_path = save_path_full
            else:
                continue

        file = h5py.File(save_path, 'r')
        dset = file['attention_scores']
        coord_dset = file['coords']
        scores = dset[:]
        coords = coord_dset[:]
        file.close()

        heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': heatmap_args.vis_level,
                            'blur': heatmap_args.blur, 'custom_downsample': heatmap_args.custom_downsample}
        if heatmap_args.use_ref_scores:
            heatmap_vis_args['convert_to_percentiles'] = False

        heatmap_save_name = '{}_{}_roi_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(slide_id, float(
            patch_args.overlap), int(heatmap_args.use_roi),
                                                                                            int(heatmap_args.blur),
                                                                                            int(heatmap_args.use_ref_scores),
                                                                                            int(heatmap_args.blank_canvas),
                                                                                            float(
                                                                                                heatmap_args.alpha),
                                                                                            int(heatmap_args.vis_level),
                                                                                            int(heatmap_args.binarize),
                                                                                            float(
                                                                                            heatmap_args.binary_thresh),
                                                                                            heatmap_args.save_ext)

        if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
            pass

        else:
            heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object,
                                  cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, **heatmap_vis_args,
                                  binarize=heatmap_args.binarize,
                                  blank_canvas=heatmap_args.blank_canvas,
                                  thresh=heatmap_args.binary_thresh, patch_size=vis_patch_size,
                                  overlap=patch_args.overlap,
                                  top_left=top_left, bot_right=bot_right)
            if heatmap_args.save_ext == 'jpg':
                heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
            else:
                heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))

        if heatmap_args.save_orig:
            if heatmap_args.vis_level >= 0:
                vis_level = heatmap_args.vis_level
            else:
                vis_level = vis_params['vis_level']
            heatmap_save_name = '{}_orig_{}.{}'.format(slide_id, int(vis_level), heatmap_args.save_ext)
            if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
                pass
            else:
                heatmap = wsi_object.visWSI(vis_level=vis_level, view_slide_only=True,
                                            custom_downsample=heatmap_args.custom_downsample)
                if heatmap_args.save_ext == 'jpg':
                    heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
                else:
                    heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))

    with open(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, 'config.yaml'), 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False)
