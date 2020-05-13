import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()
import os
import time
import argparse
from dense_correspondence.evaluation.evaluation import DenseCorrespondenceEvaluation
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset

def parse_args():
    parser = argparse.ArgumentParser(
        description='Runs all the quantitative evaluations on the trained model')

    parser.add_argument('--model_dir', type=str,
                        help='relative path to a folder of models',
                        default='trained_models/new_test_caterpillar')
    parser.add_argument('--num_image_pairs', type=int,
                        help='number of image pairs to be evaulated', default=100)
    return parser.parse_args()

def evaulate_model(model_lst, num_image_pairs, gt_dataset_config):

    gt_dataset = SpartanDataset(config_expanded=gt_dataset_config)
    
    num_image_pairs = num_image_pairs
    
    DCE = DenseCorrespondenceEvaluation

    for subdir in model_lst:
        print("evaluate model {} on dataset {}".format(subdir, gt_dataset_config['logs_root_path']))
        start_time = time.time()
        DCE.run_evaluation_on_network(model_folder=subdir, num_image_pairs=num_image_pairs,dataset=gt_dataset)
        end_time = time.time()
        print("evaluation takes %.2f seconds" %(end_time - start_time))

if __name__ == '__main__':
    args = parse_args()
    model_dir = args.model_dir
    num_image_pairs = args.num_image_pairs
    gt_dataset_config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config','dense_correspondence', 'evaluation', 'gt_dataset.yaml')
    # gt_dataset_config_file = '/home/kudo/data/pdc/trained_models/new_test_caterpillar/default_scaling_gt_pose_3/dataset.yaml'
    gt_dataset_config = utils.getDictFromYamlFilename(gt_dataset_config_file)

    abs_model_dir = utils.convert_data_relative_path_to_absolute_path(model_dir)
    d = abs_model_dir
    model_lst = [model_dir + '/' + o for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
    print("model list")
    print(model_lst)
    evaulate_model(model_lst, num_image_pairs, gt_dataset_config)