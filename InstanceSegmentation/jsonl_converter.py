import argparse
import os
import json
import torch
import numpy as np
import PIL.Image as Image
import xml.etree.ElementTree as ET

from azureml.contrib.automl.dnn.vision.object_detection.common.masktools import convert_mask_to_polygon


def binarise_mask(mask_fname):

    mask = Image.open(mask_fname)
    mask = np.array(mask)
    # instances are encoded as different colors
    obj_ids = np.unique(mask)
    # first id is the background, so remove it
    obj_ids = obj_ids[1:]

    # split the color-encoded mask into a set of binary masks
    binary_masks = mask == obj_ids[:, None, None]
    return binary_masks


def parsing_mask(mask_fname):

    # For this particular dataset, initially each mask was merged (based on binary mask of each object)
    # in the order of the bounding boxes described in the corresponding PASCAL VOC annotation file.
    # Therefore, we have to extract each binary mask which is in the order of objects in the annotation file.
    # https://github.com/microsoft/computervision-recipes/blob/master/utils_cv/detection/dataset.py
    binary_masks = binarise_mask(mask_fname)
    polygons = []
    for bi_mask in binary_masks:

        mask_tensor = torch.from_numpy(bi_mask)
        if len(mask_tensor.shape) == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        polygon = convert_mask_to_polygon(mask_tensor)
        polygons.append(polygon)

    return polygons


def convert_mask_in_VOC_to_jsonl(base_dir):

    src = base_dir #"./odFridgeObjectsMask/"
    train_validation_ratio = 5

    # Retrieving default datastore that got automatically created when we setup a workspace
    workspaceblobstore = 'workspaceblobstore' #ws.get_default_datastore().name

    # Path to the annotations
    annotations_folder = os.path.join(src, "annotations")
    mask_folder = os.path.join(src, "segmentation-masks")

    # Path to the training and validation files
    train_annotations_file = os.path.join(src, "train_annotations.jsonl")
    validation_annotations_file = os.path.join(src, "validation_annotations.jsonl")

    # sample json line dictionary
    json_line_sample = \
        {
            "image_url": "AmlDatastore://" + workspaceblobstore + "/"
                         + os.path.basename(os.path.dirname(src)) + "/" + "images",
            "image_details": {"format": None, "width": None, "height": None},
            "label": []
        }

    # Read each annotation and convert it to jsonl line
    with open(train_annotations_file, 'w') as train_f:
        with open(validation_annotations_file, 'w') as validation_f:
            for i, filename in enumerate(os.listdir(annotations_folder)):
                if filename.endswith(".xml"):
                    print("Parsing " + os.path.join(src, filename))

                    root = ET.parse(os.path.join(annotations_folder, filename)).getroot()

                    width = int(root.find('size/width').text)
                    height = int(root.find('size/height').text)
                    # convert mask into polygon
                    mask_fname = os.path.join(mask_folder, filename[:-4] + ".png")
                    polygons = parsing_mask(mask_fname)

                    labels = []
                    for index, object in enumerate(root.findall('object')):
                        name = object.find('name').text
                        isCrowd = int(object.find('difficult').text)
                        labels.append({"label": name,
                                       "bbox": "null",
                                       "isCrowd": isCrowd,
                                       'polygon': polygons[index]})

                    # build the jsonl file
                    image_filename = root.find("filename").text
                    _, file_extension = os.path.splitext(image_filename)
                    json_line = dict(json_line_sample)
                    json_line["image_url"] = json_line["image_url"] + "/" + image_filename
                    json_line["image_details"]["format"] = file_extension[1:]
                    json_line["image_details"]["width"] = width
                    json_line["image_details"]["height"] = height
                    json_line["label"] = labels

                    if i % train_validation_ratio == 0:
                        # validation annotation
                        validation_f.write(json.dumps(json_line) + "\n")
                    else:
                        # train annotation
                        train_f.write(json.dumps(json_line) + "\n")
                else:
                    print("Skipping unknown file: {}".format(filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data_path", type=str, help="the directory contains images, annotations, and masks")

    args, remaining_args = parser.parse_known_args()
    data_path = args.data_path
    odFridgeObjectsMask(data_path)


