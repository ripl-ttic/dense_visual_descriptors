import dense_correspondence_manipulation.utils.utils as utils
from depth_generation import generate_depth_images
from replace_poses import replace_poses
from evaluate_script import evaulate_model
from training_script import pdc_train

import argparse
import shutil
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description='Do essential works to connect monodepth pipeline and pdc pipeline')

    parser.add_argument('--config_path', type=str,
                        help='relative path to the config file',
                        default='config/pipe_joint_config.yaml')

    return parser.parse_args()


class PipeJoint:


    def __init__(self, config_path):
        # manuly write configs, debug use only
        self.config = {}
        self.config['generate_dataset'] = {}
        self.config['generate_dataset']['required'] = False
        self.config['generate_dataset']['generate_depth'] = True
        self.config['generate_dataset']['data_source'] = '../data/pdc'
        self.config['generate_dataset']['original_data'] = 'logs_proto_original'
        self.config['generate_dataset']['meta_dir'] = None
        self.config['generate_dataset']['image_dir'] = None
        self.config['generate_dataset']['output_dir'] = None
        self.config['generate_dataset']['model_path'] = '../data/pdc/depth_models/weights_199'
        self.config['generate_dataset']['scaling_method'] = 'default_scaling'
        self.config['generate_dataset']['ext'] = 'png'
        self.config['generate_dataset']['no_cuda'] = False
        self.config['generate_dataset']['replace_poses'] = False
        self.config['generate_dataset']['pose_data_path'] = '../data/pdc/poses'
        
        self.config['train'] = {}
        self.config['train']['required'] = True
        self.config['train']['dataset_config_file'] = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 
                               'dataset', 'composite', 'caterpillar_upright.yaml')
        self.config['train']['train_config_file'] = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 
                                'training', 'training.yaml')
        self.config['train']['logging_dir'] = "trained_models/new_test_caterpillar/"
        self.config['train']['num_iterations'] = (1500/4)-1
        self.config['train']['dimension'] = 3
        self.config['train']['dataset'] = None

        self.config['evaluate'] = {}
        self.config['evaluate']['required'] = True

        # init start
        self.config_path = config_path
        # self.config = utils.getDictFromYamlFilename(config_path)

        if self.config['generate_dataset']['required']:
            self.generate_dataset_required = True
            if self.config['generate_dataset']['generate_depth']:
                self.generate_depth_required = True
                self.data_source = self.config['generate_dataset']['data_source']
                self.original_data = self.config['generate_dataset']['original_data']
                self.meta_dir = self.config['generate_dataset']['meta_dir']
                self.image_dir = self.config['generate_dataset']['image_dir']
                self.output_dir = self.config['generate_dataset']['output_dir']
                self.model_path = self.config['generate_dataset']['model_path']
                self.scaling_method = self.config['generate_dataset']['scaling_method']
                self.ext = self.config['generate_dataset']['ext']
                self.no_cuda = self.config['generate_dataset']['no_cuda']
            else:
                self.generate_depth_required = False

            if self.config['generate_dataset']['replace_poses']:
                self.replace_poses_required = True
                self.pose_data_path = self.config['generate_dataset']['replace_poses']['pose_data_path']
            else:
                self.replace_poses_required = False

            # set up datset name
            self.dataset_name = 'logs_proto_'
            if self.generate_depth_required:
                self.dataset_name += self.scaling_method + '_'
            else:
                self.dataset_name += 'gt_depth_'

            if self.replace_poses_required:
                self.dataset_name += 'pred_pose'
            else:
                self.dataset_name += 'gt_pose'

            if self.meta_dir is None:
                self.meta_dir = os.path.join(self.data_source, self.dataset_name)
        else:
            self.generate_dataset_required = False
        
        if self.config['train']['required']:
            self.train_required = True
            self.train_dataset_config_file = self.config['train']['dataset_config_file']
            self.train_config_file = self.config['train']['train_config_file']
            self.train_logging_dir = self.config['train']['logging_dir']
            self.train_num_iterations = self.config['train']['num_iterations']
            self.train_dimension = self.config['train']['dimension']
            self.train_dataset = ''
            if self.generate_dataset_required:
                self.train_dataset = self.dataset_name
            if self.config['train']['dataset'] is not None:
                self.train_dataset = self.config['train']['dataset']
            
            

        if self.config['evaluate']['required']:
            self.evaluate_required = True

    def get_config(self, verbose=False):
        if verbose:
            print(self.config)
        return self.config


    def save_config(self):
        utils.saveToYaml(self.config, self.config_path)


    def execute(self):
        # data generation phase
        if self.generate_dataset_required:
            if self.generate_depth_required:
                src = os.path.join(self.data_source, self.original_data)
                dest = self.meta_dir
                if not os.path.isdir(dest):
                    print("start copying data")
                    start_time = time.time()
                    shutil.copytree(src, dest)
                    end_time = time.time()
                    print("data copying using {}s".format(end_time - start_time))

                print("start generating depth images")
                start_time = time.time()
                generate_depth_images(self.model_path, self.scaling_method, 
                    self.ext, self.no_cuda, self.meta_dir, self.image_dir, self.output_dir)
                end_time = time.time()
                print('generating depth images using {}s'.format(end_time - start_time))

            if self.replace_poses_required:
                self.logs_proto_path = self.meta_dir
                print('start replacing poses')
                start_time = time.time()
                replace_poses(self.pose_data_path, self.logs_proto_path)
                end_time = time.time()
                print('replacing poses takes {}s'.format(end_time - start_time))

        
        # training phase
        if self.train_required:
            train_dataset_config = utils.getDictFromYamlFilename(self.train_dataset_config_file)
            train_dataset_config['logs_root_path'] = self.train_dataset
            train_config = utils.getDictFromYamlFilename(self.train_config_file)

            pdc_train(train_dataset_config, train_config, 
                self.train_logging_dir, self.train_num_iterations, self.train_dimension)
        
        
        # evaluation phase
        # if self.evaluate_required:
        #     evaulate_model()

if __name__ == '__main__':
    args = parse_args()
    pipe_joint = PipeJoint(args.config_path)
    pipe_joint.save_config()
    pipe_joint.get_config(verbose=True)
    pipe_joint.execute()


# TODOs:
# 1. automate data copy and dataset naming
# 2. experiment config save
# 3. objectize train module
# 4. objectize evalute module
# 5. quantitative evaluation