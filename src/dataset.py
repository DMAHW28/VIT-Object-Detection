import os
from sklearn.model_selection import train_test_split

project_path = os.path.abspath('.')
dir_images_cls = ["airplanes", "helicopter", "Motorbikes"]
dir_annot_cls = ["Airplanes_Side_2", "helicopter", "Motorbikes_16"]

path_images = os.path.join(project_path, "database/101_ObjectCategories")
path_annots = os.path.join(project_path, "database/Annotations")
COLS = ["image_file", "boxe_file", "cls_name"]

def create_txt_data_torch_format(test_size=0.2):

    images_paths_images = []
    images_paths_annots = []
    images_paths_cls = []

    for dir_cls, dir_annot in zip(dir_images_cls, dir_annot_cls):
        path_img = os.path.join(path_images, dir_cls)
        path_annot = os.path.join(path_annots, dir_annot)

        img_path = [f"{path_img}/{f}" for f in os.listdir(path_img) if os.path.isfile(os.path.join(path_img, f))]
        annot_path = [f"{path_annot}/{f}" for f in os.listdir(path_annot) if os.path.isfile(os.path.join(path_annot, f))]
        img_path.sort()
        annot_path.sort()
        cls = [dir_cls] * len(img_path)
        assert len(img_path) == len(annot_path) == len(cls)
        images_paths_images.extend(img_path)
        images_paths_annots.extend(annot_path)
        images_paths_cls.extend(cls)

    train_imgs_file, test_imgs_file, train_boxes_file, test_boxes_file, train_cls_file, test_cls_file = train_test_split(images_paths_images, images_paths_annots, images_paths_cls, test_size=test_size, random_state=42)

    assert len(train_imgs_file) == len(train_boxes_file) == len(train_cls_file)
    assert len(test_imgs_file) == len(test_boxes_file) == len(test_cls_file)

    create_txt_dataset((train_imgs_file, train_boxes_file, train_cls_file), os.path.join(project_path, "database/train.txt"))
    create_txt_dataset((test_imgs_file, test_boxes_file, test_cls_file), os.path.join(project_path, "database/test.txt"))

def create_txt_dataset(data, dataset_file):
    dataset_imgs, dataset_boxes, dataset_cls = data
    with open(dataset_file, "w") as f:
        f.write(",".join(col for col in COLS) + '\n')
        for sample in zip(dataset_imgs, dataset_boxes, dataset_cls):
            line = ','.join(str(inf) for inf in sample)
            f.write(line + '\n')
        f.close()

create_txt_data_torch_format(0.2)