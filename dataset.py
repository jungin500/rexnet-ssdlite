from xml.etree import ElementTree as ET
from torch.utils.data import Dataset
from loguru import logger
from tqdm import tqdm
import numpy as np
import torch

import os
import cv2
import glob

def parse_annotation(data_root, folds, num_classes):
    fold_item_list = open(os.path.join(data_root, 'ImageSets', 'Main', folds + '.txt')).read().splitlines()
    
    xml_root_path = os.path.join(data_root, 'Annotations')
    jpeg_root_path = os.path.join(data_root, 'JPEGImages')

    xml_file_list = [os.path.join(xml_root_path, k + '.xml') for k in fold_item_list]
    assert len(xml_file_list) != 0, "No annotations in \"%s\" folder!" % data_root

    parsed_annotations = []
    label_map = {}
    for xml_file in tqdm(xml_file_list, desc='Reading annotation list'):
        elem = ET.parse(xml_file)  # ET.root
        root = elem.getroot()  # <annotation />

        filename = root.find('filename').text
        if '.jp' not in filename:
            filename = '%s.jpg' % filename
            
        boxes = []
        labels = []
        size_dom = root.find('size')
        img_width, img_height = int(size_dom.find('width').text), int(size_dom.find('height').text)
        objects = root.findall('object')
        for object_dom in objects:
            object_name = object_dom.find('name').text
            bndbox_dom = object_dom.find('bndbox')
            xmin, ymin, xmax, ymax = [int(bndbox_dom.find(k).text) for k in ['xmin', 'ymin', 'xmax', 'ymax']]

            boxes.append(np.array([xmin, ymin, xmax, ymax]))
            labels.append(object_name)

            label_map[object_name] = True

        image_path = os.path.join(jpeg_root_path, filename)
        parsed_annotations.append((
            image_path
            , {
                'boxes': torch.from_numpy(np.array(boxes)).float(),
                'labels': labels
            }
        ))

    # Label string -> int
    label_name_list = list(label_map.keys())
    label_name_list.sort()
    label_name_map = { label_name_list[idx]: idx for idx in range(len(label_name_list)) }
    logger.warning('IMPORTANT: Label list:' + str(label_name_map))
    assert num_classes == len(label_name_map.keys()), "Label count %d doesn't match as you gave: %d classes!" % (len(label_name_map.keys()), num_classes)
        

    for image_path, annotation in parsed_annotations:
        annotation['labels'] = torch.from_numpy(
            np.array(list(map(lambda label_name: label_name_map[label_name], annotation['labels'])))
        ).long()

    return parsed_annotations
        

class VOCDetectionDataset(Dataset):
    def __init__(
        self,
        data_root,
        folds,
        num_classes,
        class_map,
        half = False,
        transform = None,
        size = (224, 224)
    ):
        self.size = size
        self.half = half
        self.data_root = data_root
        self.num_classes = num_classes
        self.class_map = class_map
        self.transform = transform
        self.items = parse_annotation(data_root, folds, num_classes)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, annotation = self.items[idx]
        image = cv2.imread(img_path)
        image = cv2.resize(image, self.size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        if self.transform is not None:
            tensor = self.transform(image)
        else:
            tensor = np.transpose(image, (2, 0, 1))  # HWC -> CHW
            tensor = torch.from_numpy(tensor)
        
        tensor = tensor.float()
        if self.half:
            tensor = tensor.half()

        return tensor, annotation