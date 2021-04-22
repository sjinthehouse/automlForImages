import json
import os
import sys
import argparse

from azureml.core import Dataset, Run
from azureml.contrib.dataset.labeled_dataset import _LabeledDatasetFactory, LabeledDatasetTask

# Define Converters

class CocoToJSONLinesConverter:
    def convert(self): raise NotImplementedError

class MultiClassConverter(CocoToJSONLinesConverter):

    def __init__(self, coco_data):
        self.json_lines_data = []
        self.categories = {}
        self.coco_data = coco_data
        for i in range(0, len(coco_data['images'])):
            self.json_lines_data.append({})
        for i in range(0, len(coco_data['categories'])):
            self.categories[coco_data['categories'][i]['id']] = coco_data['categories'][i]['name']

    def _populate_image_url(self, json_line, coco_image_url):
        json_line['image_url'] = coco_image_url
        return json_line

    def _populate_label(self, json_line, label_id):
        json_line['label'] = self.categories[label_id]
        return json_line

    def _populate_label_confidence(self, json_line):
        json_line['label_confidence'] = 1.0
        return json_line

    def convert(self):
        for i in range(0, len(self.coco_data['images'])):
            self.json_lines_data[i] = {}
            self.json_lines_data[i] = self._populate_image_url(self.json_lines_data[i], self.coco_data['images'][i]['coco_url'])
            self.json_lines_data[i] = self._populate_label(self.json_lines_data[i], self.coco_data['annotations'][i]['category_id'])
            self.json_lines_data[i] = self._populate_label_confidence(self.json_lines_data[i])
        return self.json_lines_data

class MultiLabelConverter(CocoToJSONLinesConverter):

    def __init__(self, coco_data):
        self.json_lines_data = []
        self.categories = {}
        self.coco_data = coco_data
        self.image_id_to_data_index = {}
        for i in range(0, len(coco_data['images'])):
            self.json_lines_data.append({})
            self.json_lines_data[i]['image_url'] = ""
            self.json_lines_data[i]['label'] = []
            self.json_lines_data[i]['label_confidence'] = []
        for i in range(0, len(coco_data['categories'])):
            self.categories[coco_data['categories'][i]['id']] = coco_data['categories'][i]['name']

    def _populate_image_url(self, index, coco_image):
        self.json_lines_data[index]['image_url'] = coco_image['coco_url']
        self.image_id_to_data_index[coco_image['id']] = index

    def _populate_label(self, annotation):
        index = self.image_id_to_data_index[annotation['image_id']]
        self.json_lines_data[index]['label'].append(self.categories[annotation['category_id']])
        self._populate_label_confidence(index)

    def _populate_label_confidence(self, index):
        self.json_lines_data[index]['label_confidence'].append(1.0)

    def convert(self):
        for i in range(0, len(self.coco_data['images'])):
            self._populate_image_url(i, self.coco_data['images'][i])
        for i in range(0, len(self.coco_data['annotations'])):
            self._populate_label(self.coco_data['annotations'][i])
        return self.json_lines_data

class BoundingBoxConverter(CocoToJSONLinesConverter):

    def __init__(self, coco_data):
        self.json_lines_data = []
        self.categories = {}
        self.coco_data = coco_data
        self.image_id_to_data_index = {}
        for i in range(0, len(coco_data['images'])):
            self.json_lines_data.append({})
            self.json_lines_data[i]['image_url'] = ""
            self.json_lines_data[i]['label'] = []
            self.json_lines_data[i]['label_confidence'] = []
        for i in range(0, len(coco_data['categories'])):
            self.categories[coco_data['categories'][i]['id']] = coco_data['categories'][i]['name']

    def _populate_image_url(self, index, coco_image):
        self.json_lines_data[index]['image_url'] = coco_image['coco_url']
        self.image_id_to_data_index[coco_image['id']] = index

    def _populate_bbox_in_label(self, label, annotation):
        label['topX'] = annotation['bbox'][0]
        label['topY'] = annotation['bbox'][1]
        label['bottomX'] = annotation['bbox'][0] + annotation['bbox'][2]
        label['bottomY'] = annotation['bbox'][1] + annotation['bbox'][3]

    def _populate_label(self, annotation):
        index = self.image_id_to_data_index[annotation['image_id']]
        label = {'label': self.categories[annotation['category_id']]}
        self._populate_bbox_in_label(label, annotation)
        self.json_lines_data[index]['label'].append(label)
        self._populate_label_confidence(index)

    def _populate_label_confidence(self, index):
        self.json_lines_data[index]['label_confidence'].append(1.0)

    def convert(self):
        for i in range(0, len(self.coco_data['images'])):
            self._populate_image_url(i, self.coco_data['images'][i])
        for i in range(0, len(self.coco_data['annotations'])):
            self._populate_label(self.coco_data['annotations'][i])
        return self.json_lines_data

class PolygonConverter(CocoToJSONLinesConverter):

    def __init__(self, coco_data):
        self.json_lines_data = []
        self.categories = {}
        self.coco_data = coco_data
        self.image_id_to_data_index = {}
        for i in range(0, len(coco_data['images'])):
            self.json_lines_data.append({})
            self.json_lines_data[i]['image_url'] = ""
            self.json_lines_data[i]['label'] = []
            self.json_lines_data[i]['label_confidence'] = []
        for i in range(0, len(coco_data['categories'])):
            self.categories[coco_data['categories'][i]['id']] = coco_data['categories'][i]['name']

    def _populate_image_url(self, index, coco_image):
        self.json_lines_data[index]['image_url'] = coco_image['coco_url']
        self.image_id_to_data_index[coco_image['id']] = index

    def _populate_bbox_in_label(self, label, annotation):
        top_x = annotation['bbox'][0]
        top_y = annotation['bbox'][1]
        bottom_x = annotation['bbox'][0] + annotation['bbox'][2]
        bottom_y = annotation['bbox'][1] + annotation['bbox'][3]
        label['bbox'] = [top_x, top_y, bottom_x, bottom_y]

    def _populate_label(self, annotation):
        index = self.image_id_to_data_index[annotation['image_id']]
        label = {'label': self.categories[annotation['category_id']], 'isCrowd': False,
                 'polygon': annotation['segmentation']}
        self._populate_bbox_in_label(label, annotation)
        self.json_lines_data[index]['label'].append(label)
        self._populate_label_confidence(index)

    def _populate_label_confidence(self, index):
        self.json_lines_data[index]['label_confidence'].append(1.0)

    def convert(self):
        for i in range(0, len(self.coco_data['images'])):
            self._populate_image_url(i, self.coco_data['images'][i])
        for i in range(0, len(self.coco_data['annotations'])):
            self._populate_label(self.coco_data['annotations'][i])
        return self.json_lines_data

if __name__ == "__main__":
    # Parse arguments that are passed into the script
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_coco_file_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--output_file_name', type=str, required=True)
    parser.add_argument('--task_type', type=str, required=True,
        choices=[ 'ImageClassification', 'InstanceSegmentation', 'ImageMultiLabelClassification', 'ObjectDetection'])
    parser.add_argument('--base_url', type=str, default=None)

    args=parser.parse_args()

    input_coco_file_path = args.input_coco_file_path
    output_dir = args.output_dir
    output_file_path = output_dir + "/" + args.output_file_name
    task_type = args.task_type
    base_url = args.base_url


    def read_coco_file(coco_file):
        with open(coco_file) as f_in:
            return json.load(f_in)

    def write_json_lines(converter, filename, base_url=None):
        json_lines_data = converter.convert()
        with open(filename, 'w') as outfile:
            for json_line in json_lines_data:
                if base_url is not None:
                    image_url =  json_line["image_url"]
                    json_line["image_url"] = base_url + image_url[image_url.rfind("/")+1:]
                json.dump(json_line, outfile, separators=(',', ':'))
                outfile.write('\n')
            print(f"Conversion completed. Converted {len(json_lines_data)} lines.")

    coco_data = read_coco_file(input_coco_file_path)

    print("Converting for {}".format(task_type))

    if task_type == LabeledDatasetTask.IMAGE_CLASSIFICATION.value:
        converter = MultiClassConverter(coco_data)
        write_json_lines(converter, output_file_path, base_url)

    elif task_type == LabeledDatasetTask.IMAGE_MULTI_LABEL_CLASSIFICATION.value:
        converter = MultiLabelConverter(coco_data)
        write_json_lines(converter, output_file_path, base_url)

    elif task_type == LabeledDatasetTask.OBJECT_DETECTION.value:
        converter = BoundingBoxConverter(coco_data)
        write_json_lines(converter, output_file_path, base_url)

    elif task_type == LabeledDatasetTask.IMAGE_INSTANCE_SEGMENTATION.value:
        converter = PolygonConverter(coco_data)
        write_json_lines(converter, output_file_path, base_url)

    else:
        print("ERROR: Invalid Task Type")
        pass
       