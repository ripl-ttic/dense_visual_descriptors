import os

meta_dir = "/media/kudo/Elements/data/pdc/experiments/exp_05262020-051440/logs_proto_unit_scaling_gt_pose"
d = meta_dir
count = 0
folders = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
for folder in folders:
    image_dir = os.path.join(folder, "processed", "rendered_images")
    for image_file in os.listdir(image_dir):
        # print(image_file)
        # count += 1
        [image_name,image_ext] = image_file.split('.')
        # print(image_name[-5:])
        # print(image_ext)
        if image_name[-5:] == 'depth':
            count += 1
            src = os.path.join(image_dir, image_file)
            dest = os.path.join(image_dir, image_name + '_hidden' + '.' + image_ext)
            os.rename(src, dest)

print(count)