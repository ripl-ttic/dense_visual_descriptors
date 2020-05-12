import os
import shutil

# copy and replace poses
def replace_poses(pose_folder_path, logs_proto_path):
    pose_files = os.listdir(pose_folder_path)
    for pose_file in pose_files:
        if pose_file[-4:] == "yaml":
            pose_date = pose_file[:-5]

            target_pose_path = os.path.join(logs_proto_path,pose_date,"processed/images/pose_data.yaml")
            shutil.copyfile(pose_file,target_pose_path)
    print("Finish replacing original poses with predicted poses.")

if __name__ == '__main__':
    # set up paths
    logs_proto_path = '/media/kudo/Elements/data/pdc/logs_proto'
    pose_folder_path = '.'