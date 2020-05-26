import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()
from depth_generation import generate_depth_images
from replace_poses import replace_poses
from evaluate_script import evaulate_model
from training_script import pdc_train

import argparse
import shutil
import os
import time
from datetime import datetime


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
        self.config['generate_dataset']['output_target'] = 'rendered_images'
        self.config['generate_dataset']['image_dir'] = None
        self.config['generate_dataset']['output_dir'] = None
        self.config['generate_dataset']['model_path'] = '../data/pdc/depth_models/weights_199'
        self.config['generate_dataset']['scaling_method'] = 'default_scaling'
        self.config['generate_dataset']['zero_masked'] = False
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
        self.config['train']['logging_dir'] = "trained_models/new_new_test_caterpillar/"
        self.config['train']['num_iterations'] = (1500/4)-1
        self.config['train']['dimension'] = 3
        # self.config['train']['dataset'] = "logs_proto_original"
        self.config['train']['dataset'] = "experiments/exp_05262020-174446/logs_proto_default_scaling_gt_pose"
        # self.config['train']['dataset'] = "logs_proto_unit_scaling_gt_pose"
        # self.config['train']['dataset'] = "logs_proto_default_scaling_gt_pose"
        # self.config['train']['dataset'] = None

        self.config['evaluate'] = {}
        self.config['evaluate']['required'] = False
        self.config['evaluate']['model_lst'] = ['trained_models/new_test_caterpillar/default_scaling_gt_pose_3','trained_models/new_test_caterpillar/original_3','trained_models/new_test_caterpillar/unit_scaling_gt_pose_3']
        self.config['evaluate']['num_image_pairs'] = 100
        self.config['evaluate']['gt_dataset_config_file'] = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config','dense_correspondence', 'evaluation', 'gt_dataset.yaml')

        self.config['experiments'] = {}
        self.config['experiments']['as_experiment'] = True

        # init start
        self.config_path = config_path
        # self.config = utils.getDictFromYamlFilename(config_path)

        # configs about experiment phase
        self.as_experiment = self.config['experiments']['as_experiment']
        # create experiment folder if needed
        if self.as_experiment:
            self.current_timestamp = datetime.now().strftime("%m%d%Y-%H%M%S")
            self.exp_dir = 'experiments/exp_' + self.current_timestamp

        # configs about dataset generation
        if not self.config['generate_dataset']['required']:
            self.generate_dataset_required = False
        else:
            self.generate_dataset_required = True
            if self.config['generate_dataset']['generate_depth']:
                self.generate_depth_required = True
                self.data_source = self.config['generate_dataset']['data_source']
                self.original_data = self.config['generate_dataset']['original_data']
                self.meta_dir = self.config['generate_dataset']['meta_dir']
                self.output_target = self.config['generate_dataset']['output_target']
                self.image_dir = self.config['generate_dataset']['image_dir']
                self.output_dir = self.config['generate_dataset']['output_dir']
                self.model_path = self.config['generate_dataset']['model_path']
                self.scaling_method = self.config['generate_dataset']['scaling_method']
                self.zero_masked = self.config['generate_dataset']['zero_masked']
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

            if self.as_experiment:
                self.dataset_name = self.exp_dir + '/' + self.dataset_name

            if self.meta_dir is None:
                self.meta_dir = os.path.join(self.data_source, self.dataset_name)
            
        # configs about training phase
        if not self.config['train']['required']:
            self.train_required = False
        else:
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

            if self.as_experiment:
                self.train_logging_dir = self.exp_dir + '/trained_models' 

        # configs about evaluation phase
        if not self.config['evaluate']['required']:
            self.evaluate_required = False
        else:
            self.evaluate_required = True
            self.eval_model_lst = self.config['evaluate']['model_lst']
            self.eval_num_image_pairs = self.config['evaluate']['num_image_pairs'] = 100
            self.eval_gt_dataset_config_file = self.config['evaluate']['gt_dataset_config_file']


    def get_config(self, verbose=False):
        if verbose:
            print(self.config)
        return self.config


    def save_config(self, config_path=None):
        if config_path is None:
            utils.saveToYaml(self.config, self.config_path)
        else:
            utils.saveToYaml(self.config, config_path)


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
                generate_depth_images(self.model_path, self.scaling_method, self.zero_masked,
                    self.ext, self.no_cuda, self.meta_dir, self.output_target, self.image_dir, self.output_dir)
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

            pdc_train(train_dataset_config, train_config, self.train_dataset[11:],
                self.train_logging_dir, self.train_num_iterations, self.train_dimension)
        
        # evaluation phase
        if self.evaluate_required:
            if not self.as_experiment:
                evaulate_model(model_lst=self.eval_model_lst, 
                    num_image_pairs=self.eval_num_image_pairs)
            else:
                # experiment mode
                self.gt_dataset_config = utils.getDictFromYamlFilename(self.eval_gt_dataset_config_file)
                
                evaulate_model(model_lst=self.eval_model_lst, output_dir=self.exp_dir,
                    num_image_pairs=self.eval_num_image_pairs, 
                    gt_dataset_config=self.gt_dataset_config)     
                
        # save experiment config        
        if self.as_experiment:
            experiment_config_path = os.path.join(utils.get_data_dir(), self.exp_dir, 'config.yaml')
            self.save_config(experiment_config_path)


if __name__ == '__main__':
    args = parse_args()
    pipe_joint = PipeJoint(args.config_path)
    pipe_joint.save_config() 
    pipe_joint.get_config(verbose=True)
    pipe_joint.execute()


# TODOs:
# 1. improve save_log function 
# 2. readme on how to use
# 3. quantitative evaluation