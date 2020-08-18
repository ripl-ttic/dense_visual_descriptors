import os
import shutil

orig_dir = "/media/kudo/Elements/data/pdcized_kitti/logs_proto"
left_dir = "/media/kudo/Elements/data/pdcized_kitti_left/logs_proto"
right_dir = "/media/kudo/Elements/data/pdcized_kitti_right/logs_proto"

d = orig_dir
scenes = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
num_scenes = len(scenes)

for scene_i, folder in enumerate(scenes):
    print("working on the {}/{} scene".format(scene_i+1, num_scenes))
    image_dir = os.path.join(folder, "processed", "images")
    depth_dir = os.path.join(folder, "processed", "rendered_images")

    left_image_dir = os.path.join(left_dir, str(scene_i), "processed", "images")
    left_depth_dir = os.path.join(left_dir, str(scene_i), "processed", "rendered_images")
    right_image_dir = os.path.join(right_dir, str(scene_i), "processed", "images")
    right_depth_dir = os.path.join(right_dir, str(scene_i), "processed", "rendered_images")
    
    if not os.path.exists(left_image_dir):
        os.makedirs(left_image_dir)
    if not os.path.exists(left_depth_dir):
        os.makedirs(left_depth_dir)
    if not os.path.exists(right_image_dir):
        os.makedirs(right_image_dir)
    if not os.path.exists(right_depth_dir):
        os.makedirs(right_depth_dir)

    for image_file in os.listdir(image_dir):

        [image_file_name, image_file_ext] = image_file.split('.')

        # if it comes from the left camera
        if image_file_name[5] == '2':
            src = os.path.join(image_dir, image_file)
            dest = os.path.join(left_image_dir, image_file)
            shutil.copyfile(src, dest)

        # if it comes from the right camera
        if image_file_name[5] == '3':
            src = os.path.join(image_dir, image_file)
            dest = os.path.join(right_image_dir, image_file)
            shutil.copyfile(src, dest)

    # copy yamls
    src = os.path.join(image_dir, "camera_info_2.yaml")
    dest = os.path.join(left_image_dir, "camera_info.yaml")
    shutil.copyfile(src, dest)

    src = os.path.join(image_dir, "camera_info_3.yaml")
    dest = os.path.join(right_image_dir, "camera_info.yaml")
    shutil.copyfile(src, dest)

    src = os.path.join(image_dir, "pose_data_2.yaml")
    dest = os.path.join(left_image_dir, "pose_data.yaml")
    shutil.copyfile(src, dest)

    src = os.path.join(image_dir, "pose_data_3.yaml")
    dest = os.path.join(right_image_dir, "pose_data.yaml")
    shutil.copyfile(src, dest)

    
    for depth_file in os.listdir(depth_dir):

        [depth_file_name, depth_file_ext] = depth_file.split('.')

        # if it comes from the left camera
        if depth_file_name[5] == '2':
            src = os.path.join(depth_dir, depth_file)
            dest = os.path.join(left_depth_dir, depth_file)
            shutil.copyfile(src, dest)

        # if it comes from the right camera
        if depth_file_name[5] == '3':
            src = os.path.join(depth_dir, depth_file)
            dest = os.path.join(right_depth_dir, depth_file)
            shutil.copyfile(src, dest)