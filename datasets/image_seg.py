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
    image_seg_list = []
    readlist = open(imglist).readlines()
    print("Totally {} samples".format(len(readlist)))
    print("Starting Checking image&label pair list...")
    for line in readlist:
        line = line.strip()
        linesp = line.split(' ')
        if len(linesp) != 2:
            raise (RuntimeError("Image list file read line error : " + line + "\n"))
        imgname = os.path.join(root, linesp[0])
        segname = os.path.join(root, linesp[1])
        if is_image_file(imgname) and is_image_file(segname) and os.path.isfile(imgname) and os.path.isfile(segname):
            item = (imgname, segname)
            image_seg_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
    print("Checking image&label pair list done!")
    return image_seg_list


class ImageSeg(data.Dataset):
    def __init__(self, root, imglist, imgseg_transform=None):
        image_seg = make_dataset(root, imglist)
        if len(image_seg) == 0:
            raise (RuntimeError("Found 0 pair of image & seglabel in image list file: " + imglist + "\n"))
        self.root = root
        self.image_seg = image_seg
        self.imgseg_transform = imgseg_transform

    def __getitem__(self, index):
        image, seglabel = self.image_seg[index]
        img = cv2.imread(image, cv2.CV_LOAD_IMAGE_COLOR)  # BGR  3 channel ndarray wiht shape H * W * 3
        seg = cv2.imread(seglabel, cv2.CV_LOAD_IMAGE_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        if img.shape[0] != seg.shape[0] or img.shape[1] != seg.shape[1]:
            raise (RuntimeError("Image & seglabel shape mismatch: " + image + " " + seglabel + "\n"))
        if self.imgseg_transform is not None:
            img, seg = self.imgseg_transform(img, seg)
        return img, seg

    def __len__(self):
        return len(self.image_seg)
