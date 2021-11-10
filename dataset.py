from xml.etree import ElementTree as ET
from torch.utils.data import Dataset
from loguru import logger
from tqdm import tqdm
import numpy as np
import torch

import os
import cv2
import glob


def parse_annotation(data_root, folds, size, num_classes):
    fold_item_list = open(os.path.join(
        data_root, 'ImageSets', 'Main', folds + '.txt')).read().splitlines()

    xml_root_path = os.path.join(data_root, 'Annotations')
    jpeg_root_path = os.path.join(data_root, 'JPEGImages')

    xml_file_list = [os.path.join(xml_root_path, k + '.xml')
                     for k in fold_item_list]
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
        img_width, img_height = int(size_dom.find('width').text), int(
            size_dom.find('height').text)
        objects = root.findall('object')
        for object_dom in objects:
            object_name = object_dom.find('name').text
            bndbox_dom = object_dom.find('bndbox')
            xmin, ymin, xmax, ymax = [int(bndbox_dom.find(k).text) for k in [
                'xmin', 'ymin', 'xmax', 'ymax']]
            
            xmin, xmax = xmin / img_width * size[0], xmax / img_width * size[0]
            ymin, ymax = ymin / img_height * size[1], ymax / img_height * size[1]

            boxes.append(np.array([xmin, ymin, xmax, ymax]))
            labels.append(object_name)

            label_map[object_name] = True

        image_path = os.path.join(jpeg_root_path, filename)
        parsed_annotations.append((
            image_path, {
                'boxes': torch.from_numpy(np.array(boxes)).float(),
                'labels': labels
            }
        ))

    # Label string -> int
    label_name_list = list(label_map.keys())
    label_name_list.sort()
    label_name_map = {label_name_list[idx]
        : idx for idx in range(len(label_name_list))}
    logger.warning('IMPORTANT: Label list:' + str(label_name_map))
    assert num_classes == len(label_name_map.keys()), "Label count %d doesn't match as you gave: %d classes!" % (
        len(label_name_map.keys()), num_classes)

    for image_path, annotation in parsed_annotations:
        annotation['labels'] = torch.from_numpy(
            np.array(
                list(map(lambda label_name: label_name_map[label_name], annotation['labels'])))
        ).long()

    return parsed_annotations


def preload_caches(cache_filename, items, max_w, max_h):
    '''Preload image cache
    Thanks to megvii-detection/YOLOX
    (https://github.com/Megvii-BaseDetection/YOLOX/blob/abc8c870daa22faffd5eaf9d293a16d98437185a/yolox/data/datasets/voc.py#L142)
    '''
    logger.warning(
        "\n********************************************************************************\n"
        "You are using cached images in RAM to accelerate training.\n"
        "This requires large system RAM.\n"
        "Make sure you have 60G+ RAM and 19G available disk space for training VOC.\n"
        "********************************************************************************\n"
    )

    if not os.path.exists(cache_filename):
        logger.info(
            "Caching images for the first time. This might take about 3 minutes for VOC"
        )

        images = np.memmap(
            cache_filename,
            shape=(len(items), max_h, max_w, 3),
            dtype=np.uint8,
            mode='w+',
        )

        from tqdm import tqdm
        from multiprocessing.pool import ThreadPool

        def load_resized_img(idx):
            img_path, annotation = items[idx]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            assert img is not None

            r = min(max_w / img.shape[0], max_h / img.shape[1])
            resized_img = cv2.resize(
                img,
                (int(img.shape[1] * r), int(img.shape[0] * r)),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.uint8)

            return resized_img

        NUM_THREADS = min(8, os.cpu_count())
        loaded_images = ThreadPool(NUM_THREADS).imap(
            lambda x: load_resized_img(x),
            range(len(items))
        )
        pbar = tqdm(enumerate(loaded_images), total=len(items))
        for k, out in pbar:
            images[k][: out.shape[0], : out.shape[1], :] = out.copy()
        images.flush()
        pbar.close()
    else:
        logger.warning(
            "You are using cached imgs! Make sure your dataset is not changed!!\n"
            "Everytime the self.input_size is changed in your exp file, you need to delete\n"
            "the cached data and re-generate them.\n"
        )

    images = np.memmap(
        cache_filename,
        shape=(len(items), max_h, max_w, 3),
        dtype=np.uint8,
        mode='r+'
    )

    return images


class VOCDetectionDataset(Dataset):
    def __init__(
        self,
        data_root,
        folds,
        num_classes,
        class_map,
        cache=False,
        half=False,
        transform=None,
        size=(416, 416)
    ):
        self.size = size
        self.half = half
        self.data_root = data_root
        self.num_classes = num_classes
        self.class_map = class_map
        self.transform = transform
        self.cache = cache
        self.items = parse_annotation(data_root, folds, size, num_classes)
        if self.cache:
            self.images = preload_caches(
                './image_cache.array', self.items, size[0], size[1])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if self.cache:
            image = self.images[idx]
            _, annotation = self.items[idx]
        else:
            img_path, annotation = self.items[idx]
            image = cv2.imread(img_path)
            image = cv2.resize(image, self.size)
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image / 255.0).astype(np.float32)

        if self.transform is not None:
            tensor = self.transform(image)
        else:
            tensor = np.transpose(image, (2, 0, 1))  # HWC -> CHW
            tensor = torch.from_numpy(tensor)

        tensor = tensor.float()
        if self.half:
            tensor = tensor.half()

        return tensor, annotation

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from model import collate_batch
    
    logger.info("Checking dataloader")
    
    dataset = VOCDetectionDataset(
        data_root='/mnt/windows-11/Dataset/VOCdevkit/VOC2007',
        folds='trainval',
        num_classes=20,
        cache=False,
        class_map=None,
        transform=None,
        size=(416, 416)
    )
    
    trainset_items = int(len(dataset) * 0.9)
    validset_items = len(dataset) - trainset_items
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [trainset_items, validset_items])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_batch
    )
    
    image_batch, annotation_batch = next(iter(train_loader))
    print("Image length: %d, Shape: %s" % (len(image_batch), image_batch[0].shape))
    print(annotation_batch)
    
    for idx in range(len(image_batch)):
        image, annotation = image_batch[idx], annotation_batch[idx]
        
        image = image.numpy()
        image *= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        image += np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        image = np.transpose(image, (1, 2, 0))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = (image * 255.0).astype(np.uint8)
        image = cv2.resize(image, (800, 800))  # bigger size
        
        # draw annotations
        '''
        [{'boxes': tensor([[  4.,  16., 400., 400.]]), 'labels': tensor([0])}, {'boxes': tensor([[ 61.,  57., 292., 189.],
        [340.,  79., 410., 228.],
        [  1.,  32.,  72., 295.]]), 'labels': tensor([18, 14, 14])}]
        '''
        boxes, labels = annotation['boxes'], annotation['labels']
        boxes, labels = boxes.numpy(), labels.numpy()
        
        for box_idx in range(boxes.shape[0]):
            xmin, ymin, xmax, ymax = boxes[box_idx]
            xmin, ymin, xmax, ymax = [int(k / 416 * 800) for k in [xmin, ymin, xmax, ymax]]  # 416 -> 800
            label_id = labels[box_idx]
            
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        
        cv2.imshow("Image", image)
        cv2.waitKey(0)
    