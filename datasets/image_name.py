import torch.utils.data as data

import cv2
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(root, imglist):
    if not os.path.isfile(imglist):
        raise (RuntimeError("Image list file do not exist: " + imglist + "\n"))
    image_list = []
    readlist = open(imglist).readlines()
    print("Totally {} samples".format(len(readlist)))
    print("Starting Checking image list...")
    for line in readlist:
        line = line.strip()
        imgname = os.path.join(root, line)
        if is_image_file(imgname) and os.path.isfile(imgname):
            image_list.append(line)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
    print("Checking image list done!")
    return image_list


class ImageName(data.Dataset):
    def __init__(self, root, imglist, img_transform=None):
        image_list = make_dataset(root, imglist)
        if len(image_list) == 0:
            raise (RuntimeError("Found 0 image in image list file: " + imglist + "\n"))
        self.root = root
        self.image_list = image_list
        self.img_transform = img_transform

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_fullname = os.path.join(self.root, image_name)
        img = cv2.imread(image_fullname, cv2.CV_LOAD_IMAGE_COLOR)  # BGR  3 channel ndarray wiht shape H * W * 3
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img, image_name

    def __len__(self):
        return len(self.image_list)
