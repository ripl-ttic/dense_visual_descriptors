# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

from PIL import Image

import torch
from torchvision import transforms, datasets

import networks
# from utils import download_model_if_doesnt_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generating depth images from Monodepthv2 models.')

    parser.add_argument('--image_dir', type=str,
                        help='path to a folder of images')
    parser.add_argument('--meta_dir', type=str,
                        help='path to a meta folder of images')
    parser.add_argument('--output_dir', type=str,
                        help='path to save a test image or folder of images')
    # parser.add_argument('--model_name', type=str,
    #                     help='name of a pretrained model to use',
    #                     choices=[
    #                         "mono_640x192",
    #                         "stereo_640x192",
    #                         "mono+stereo_640x192",
    #                         "mono_no_pt_640x192",
    #                         "stereo_no_pt_640x192",
    #                         "mono+stereo_no_pt_640x192",
    #                         "mono_1024x320",
    #                         "stereo_1024x320",
    #                         "mono+stereo_1024x320"])
    parser.add_argument('--model_path', type=str,
                        help='path of the model to use')
    parser.add_argument('--scaling_method', type=str, 
                        help='the scaling method when converting disparity image to depth image',
                        choices=['default_scaling','unit_scaling'],
                        default='default_scaling')
    parser.add_argument('--zero_masked', 
                        help='if mask out zeros in ground truth depth image',
                        action='store_true')
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="png")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def generate_depth_images_for_a_single_folder(image_dir, output_dir, model_path, scaling_method, zero_masked, ext, no_cuda):
    """Function to predict for a single image or folder of images
    """
    # assert args.model_name is not None, \
    #     "You must specify the --model_name parameter; see README.md for an example"

    # set up min_depth and max_depth for disp_to_depth conversion. Use default values from option for now.
    
    if scaling_method == 'default_scaling':
        MIN_DEPTH = 0.1
        MAX_DEPTH = 100
    elif scaling_method == 'unit_scaling':
        MIN_DEPTH = 0.1
        MAX_DEPTH = 1
    else:
        raise ValueError('Scaling method has to be one of default_scaling or unit_scaling')

    if torch.cuda.is_available() and not no_cuda:
        device = torch.device("cuda")
        print("using cuda")
    else:
        device = torch.device("cpu")
        print("using cpu")

    # download_model_if_doesnt_exist(args.model_name)
    # model_path = os.path.join("models", args.model_name)
    model_path = model_path
    assert os.path.exists(model_path), "model path not exists"
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()
    
    output_dir = output_dir
    assert os.path.isdir(output_dir), "not a valid output directory"

    # FINDING INPUT IMAGES
    if os.path.isfile(image_dir):
        # Only testing on a single image
        paths = [image_dir]
        # output_dir = os.path.dirname(image_dir)
    elif os.path.isdir(image_dir):
        # Searching folder for images
        paths = glob.glob(os.path.join(image_dir, '*.{}'.format(ext)))
        # output_dir = image_dir
    else:
        raise Exception("Can not find image_dir: {}".format(image_dir))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            # if image_path.endswith("_disp.jpg"):
            #     # don't try to predict disparity for a disparity image!
            #     continue
                
            if image_path.endswith("_depth.png") or image_path.endswith(".yaml"):
                # don't try to predict disparity for a disparity image or a yaml file!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            # name_dest_npy = os.path.join(output_dir, "{}_disp.npy".format(output_name))
            # scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
            # np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            # disp_resized_np = disp_resized.squeeze().cpu().numpy()
            # vmax = np.percentile(disp_resized_np, 95)
            # normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            # colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            # im = pil.fromarray(colormapped_im)

            # name_dest_im = os.path.join(output_dir, "{}_disp.png".format(output_name))
            # im.save(name_dest_im)

            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            _, pred_depth_np = disp_to_depth(disp_resized_np, MIN_DEPTH, MAX_DEPTH)
            
            if zero_masked:
                # Get ground truth depth from data
                gt_depth_image_path = os.path.join(image_dir, "{}_depth.png".format(output_name[:6]))
                gt_depth_image = Image.open(gt_depth_image_path)
                gt_depth_np = np.array(gt_depth_image)
                # sorted_np_depth_lst = sorted(list(set(np_gt_depth.reshape(-1))))
            
                # non_zero_min_depth = sorted_np_depth_lst[1]
                # max_depth = sorted_np_depth_lst[-1]
            
                nonzero_mask = gt_depth_np > 0
                zero_mask = gt_depth_np == 0

                nonzero_pred_depth_np = pred_depth_np[nonzero_mask]
                nonzero_gt_depth_np = gt_depth_np[nonzero_mask]
                
                # Median Scaling
                # ratio = np.median(nonzero_gt_depth_np) / np.median(nonzero_pred_depth_np)
                # pred_depth_np *= ratio
                
                pred_depth_np = np.where(zero_mask, 0, pred_depth_np)

                # print('generated depth')
                # print(masked_pred_depth_np)
                # print(np.max(masked_pred_depth_np))
            
            # Saving single channel depth image
            im = pil.fromarray(pred_depth_np)
            im = im.convert('I')
            
            name_dest_im = os.path.join(output_dir, "{}_depth.png".format(output_name[:6]))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(paths), name_dest_im))

    print('-> Done!')
    
    
    
def generate_depth_images(model_path, scaling_method, zero_masked=False, ext='png', no_cuda=False, meta_dir=None, output_target='rendered_images',image_dir=None, output_dir=None):
    
    assert bool((image_dir is not None) and (output_dir is not None)) != bool(meta_dir is not None), "either input meta directory or (image_dir and output_dir)"
    
    # model_path = args.model_path
    # scaling_method = args.scaling_method
    # ext = args.ext
    # no_cuda = args.no_cuda
    
    if meta_dir is None:
    # if args.meta_dir is None:
        # image_dir = args.image_dir
        # output_dir = args.output_dir
        generate_depth_images_for_a_single_folder(image_dir, output_dir, model_path, ext, no_cuda)
    else:
        d = meta_dir
        # d = args.meta_dir
        folders = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]

        for folder in folders:
            image_dir = os.path.join(folder, "processed", "images")
            # output_dir = os.path.join(".","depth_output","tests")
            if output_target == 'rendered_images':
                output_dir = image_dir = os.path.join(folder, "processed", "rendered_images")
            elif output_target == 'images':
                output_dir = image_dir = os.path.join(folder, "processed", "images")
            else:
                print('Invalid output target')
            
            generate_depth_images_for_a_single_folder(image_dir, output_dir, 
                model_path, scaling_method, zero_masked, ext, no_cuda)
            

if __name__ == '__main__':
    args = parse_args()

    meta_dir = args.meta_dir
    image_dir = args.image_dir
    output_dir = args.output_dir
    model_path = args.model_path
    scaling_method = args.scaling_method
    zero_masked = args.zero_masked
    ext = args.ext
    no_cuda = args.no_cuda

    generate_depth_images(model_path, scaling_method, zero_masked, ext, no_cuda, meta_dir, image_dir, output_dir)
