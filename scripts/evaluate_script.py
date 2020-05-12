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
                        default='../data/pdc/trained_models/caterpillar')
    parser.add_argument('--num_image_pairs', type=int,
                        help='number of image pairs to be evaulated', default=100)
    return parser.parse_args()

def evaulate_model(model_dir, num_image_pairs):
    
    model_dir = utils.convert_data_relative_path_to_absolute_path(model_dir)
    num_image_pairs = num_image_pairs
    
    DCE = DenseCorrespondenceEvaluation
    d = model_dir
    subdirs = [os.path.join(d, o) for o in os.listdir(d) 
                    if os.path.isdir(os.path.join(d,o))]

    gt_dataset_config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config','dense_correspondence', 'evaluation', 'gt_dataset.yaml')
    gt_dataset_config = utils.getDictFromYamlFilename(gt_dataset_config_filename)
    gt_dataset = SpartanDataset(config_expanded=gt_dataset_config)

    print(subdirs)
    for subdir in subdirs:
        print("evaluate model {} on dataset {}".format(subdir, gt_dataset_config['logs_root_path']))
        start_time = time.time()
        DCE.run_evaluation_on_network(model_folder=subdir, num_image_pairs=num_image_pairs,dataset=gt_dataset)
        end_time = time.time()
        print("evaluation takes %.2f seconds" %(end_time - start_time))

if __name__ == '__main__':
    args = parse_args()
    model_dir = args.model_dir
    num_image_pairs = args.num_image_pairs
    evaulate_model(model_dir, num_image_pairs)