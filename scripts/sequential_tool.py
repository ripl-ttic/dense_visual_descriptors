import os
import shutil

input_dir = "/media/kudo/Elements/data/pdc/logs_proto"
output_dir = "/media/kudo/Elements/data/sequential_pdc/logs_proto"

d = input_dir
count = 0
folders = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
for folder_i, folder in enumerate(folders):
    num_folders = len(folders)
    print("working on {}/{} scene".format(folder_i+1,num_folders))
    image_dir = os.path.join(folder, "processed", "images")
    rel_path = os.path.relpath(image_dir, input_dir)
    output_image_dir = os.path.join(output_dir, rel_path)
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    rgb_lst = [filename for filename in os.listdir(image_dir) if filename.split('.')[0][-3:]=='rgb']

    for idx, image_file in enumerate(rgb_lst):

        [image_name,image_ext] = image_file.split('.')

        src = os.path.join(image_dir, image_file)
        dest = os.path.join(output_image_dir, "{:d}".format(idx).zfill(6) + '.' + image_ext)
        shutil.copyfile(src, dest)
        count += 1

    # copy yamls
    src = os.path.join(image_dir, "camera_info.yaml")
    dest = os.path.join(output_image_dir, "camera_info.yaml")
    shutil.copyfile(src, dest)

    src = os.path.join(image_dir, "pose_data.yaml")
    dest = os.path.join(output_image_dir, "pose_data.yaml")
    shutil.copyfile(src, dest)

print("The total number of rgb images copied is {}".format(count))