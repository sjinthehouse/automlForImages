# AutoML Vision

## AutoML Vision Overview

### What is AutoML Vision?
AutoML is an [Azure Machine Learning](https://azure.microsoft.com/en-us/services/machine-learning/) feature, that empowers both professional and citizen data scientists to build machine learning models rapidly. Since its launch, AutoML has helped accelerate model building for essential machine learning tasks like Classification, Regression and Time-series Forecasting. With the preview of AutoML Vision, there will be added support for Vision tasks. Data scientists will be able to easily generate models trained on image data for scenarios like Image Classification (multi-class, multi-label), Object Detection and Instance Segmentation. 

Customers across various industries are looking to leverage machine learning to build models that can process image data. Applications range from image classification of fashion photos to PPE detection in industrial environments. Customers want a solution to easily build models, controlling the model training to generate the optimal model for their training data, and a way to easily manage these ML models end-to-end. While Azure Machine Learning offers a solution for managing the end-to-end ML Lifecycle, customers currently have to rely on the tedious process of custom training their image models. Iteratively finding the right set of model algorithms and hyperparameters for these scenarios typically require significant data scientist effort.

With AutoML support for Vision tasks, Azure ML customers can easily build models trained on image data, without writing any training code. Customers can seamlessly integrate with Azure ML's Data Labeling capability and use this labeled data for generating image models. They can control the model generated by specifying the model algorithm and can optionally tune the hyperparameters. They can sweep over multiple model algorithms / hyperparameter ranges and find the optimal model for their needs. The resulting model can then be downloaded or deployed as a web service in Azure ML and can be operationalized at scale, leveraging AzureML MLOps capabilities.

Authoring AutoML models for vision tasks will be initially supported via the Azure ML Python SDK. The resulting experimentation runs, models and outputs will be accessible from the Azure ML Studio UI.

### AutoML Vision Capabilities
Azure Machine Learning is a service that accelerates the end-to-end machine learning lifecycle, helping developers and data scientists to build, train and deploy models fast, with robust MLOps capabilities to allow operationalizing these ML models at scale. AutoML Vision is a feature within Azure Machine Learning that allows users to easily and rapidly build vision models from image data, while maintaining full control and visibility over the model building process. AutoML Vision is the ideal solution for customer scenarios that might require control over model training, deployment and the end to end ML lifecycle and it is addressed to customers having machine learning knowledge in computer vision space.  
AutoML Vision includes the following feature capabilities - 
<ul>
<li>Ability to use AutoML to generate models for Image Classification, Object Detection and Instance Segmentation, via Python SDK</li>
<li>Control over training environment - model training takes place in the user's training environment, that can be secured with a virtual network. Training data never leaves the customer controlled workspace. Users can control the compute target used for training, selecting from VM SKUs with standard GPUs to  advanced multi-GPU SKUs for faster training.</li>
<li>Control over model training algorithms and hyperparameters - in some scenarios, getting optimal model performance requires tuning the underlying algorithms and hyperparameters. With AutoML, users can select specific Deep Learning architectures and customize them. This control can range from easily getting the default model for the specified architecture to advanced controls that can sweep the hyperparameter space and come up with the optimal model. When sweeping over a hyperparameter space, controls for sampling mechanisms and early termination are made available to the user, allowing for the optimal use of resource budget.</li>
<li>Control over deployment of the resulting model - AutoML models can be deployed as an endpoint in the user's AzureML workspace. Users can control the compute used for inferencing, use high performance serving with Triton Inference server and can secure the inferencing environment with a virtual network.</li>
<li>Ability to download the resulting model and use it in other environments.</li>
<li>Visibility into the model building process - users can see a leaderboard with the various configurations tried and a compare model performance using evaluation metrics and charts for each.</li>
<li>Integration with MLOps capabilities for operationalizing the resulting model at scale</li>
<li>Integration with Azure ML Data Labeling capabilities and Datasets for creating or adding to your training data</li>
<li>Integration with Azure ML Pipelines to create a workflow that stitches together various ML phases such as data preparation, training and deployment</li>
</ul>

### Target Audience
This feature is targeted to data scientists with ML knowledge in the Computer Vision space, looking to build ML models using image data in Azure Machine Learning. It targets to boost data scientist productivity, while allowing full control of the model algorithm, hyperparameters and training and deployment environments.

### Pricing
Like all Azure ML features, customers incur the costs associated with the Azure resources consumed (for example, compute and storage costs). There are no additional fees associated with Azure Machine Learning or AutoML Vision. See [Azure Machine Learning pricing](https://azure.microsoft.com/en-us/pricing/details/machine-learning/) for details.

## How to use AutoML Vision to build models for Vision tasks?
AutoML allows you to easily train models for Image Classification, Object Detection & Instance Segmentation on your image data. You can control the model algorithm to be used, specify hyperparameter values for your model as well as perform a sweep across the hyperparameter space to generate an optimal model. Parameters for configuring your AutoML Vision run are specified using the 'AutoMLVisionConfig' in the Python SDK.

### Select your task type
AutoML Vision supports the following task types -
<ul>
<li>image-classification</li>
<li>image-multi-labeling</li>
<li>image-object-detection</li>
<li>image-instance-segmentation</li>
</ul>

This task type is a required parameter and is passed in using the `task` parameter in the AutoMLVisionConfig. For example: 

```python
from azureml.train.automl import AutoMLVisionConfig
automl_vision_config = AutoMLVisionConfig(task='image-object-detection')
```

### Training and Validation data
In order to generate Vision models, you will need to bring in labeled image data as input for model training in the form of an AzureML `Labeled Dataset`. You can either use a Labeled Dataset that you have exported from a Data Labeling project, or create a new Labeled Dataset with your labeled training data. 

Labeled datasets are [Tabular datasets](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.tabulardataset)  with some enhanced capabilites such as mounting and downloading. Structure of the labeled dataset depends upon the task at hand. For Image task types, it consists of the following fields:
<ul>
<li>image_url: contains filepath as a StreamInfo object</li>
<li>image_details: image metatadata information consist of height, width and format. This field is optional and hence may or may not exist. This restriction might become mandatory in the future.</li>
<li>label: a json representation of the image label, based on the task type</li>
</ul>

Creation of labeled datasets is supported from data in JSONL format. 

#### JSONL sample schema for each task type 
Here is a sample JSONL file for Image classfication:
```python
{
      "image_url": "AmlDatastore://image_data/Image_01.png",
      "image_details":
      {
          "format": "png",
          "width": "2230px",
          "height": "4356px"
      },
      "label": "cat"
  }
  {
      "image_url": "AmlDatastore://image_data/Image_02.jpeg",
      "image_details":
      {
          "format": "jpeg",
          "width": "3456px",
          "height": "3467px"
      },
      "label": "dog"
  }
  ```
  
  And here is a sample JSONL file for Object Detection:
  ```python
  {
      "image_url": "AmlDatastore://image_data/Image_01.png",
      "image_details":
      {
          "format": "png",
          "width": "2230px",
          "height": "4356px"
      },
      "label":
      {
          "label": "cat",
          "topX": "1",
          "topY": "0",
          "bottomX": "0",
          "bottomY": "1",
          "isCrowd": "true",
      }
  }
  {
      "image_url": "AmlDatastore://image_data/Image_02.png",
      "image_details":
      {
          "format": "jpeg",
          "width": "1230px",
          "height": "2356px"
      },
      "label":
      {
          "label": "dog",
          "topX": "0",
          "topY": "1",
          "bottomX": "0",
          "bottomY": "1",
          "isCrowd": "false",
      }
  }
  ```
  

If your training data is in a different format (e.g. pascal VOC), you can leverage the helper scripts included with the sample notebooks in this repo to convert the data to JSONL. Once your data is in JSONL format, you can create a labeled dataset using this snippet -

```python
from azureml.contrib.dataset.labeled_dataset import _LabeledDatasetFactory, LabeledDatasetTask
from azureml.core import Dataset

training_dataset = _LabeledDatasetFactory.from_json_lines(
        task=LabeledDatasetTask.OBJECT_DETECTION, path=ds.path('odFridgeObjects/odFridgeObjects.jsonl'))
training_dataset = training_dataset.register(workspace=ws, name=training_dataset_name)
```

You can optionally specify another labeled dataset as a validation dataset to be used for your model. If no validation dataset is specified, 20% of your training data will be used for validation by default, unless you pass split_ratio argument with a different value.

Training data is a required parameter and is passed in using the `training_data` parameter. Validation data is optional and is passed in using the `validation_data` parameter of the AutoMLVisionConfig. For example: 

```python
from azureml.train.automl import AutoMLVisionConfig
automl_vision_config = AutoMLVisionConfig(training_data=training_dataset)
```

### Compute to run experiment
You will need to provide a [Compute Target](https://docs.microsoft.com/azure/machine-learning/service/concept-azure-machine-learning-architecture#compute-target) that will be used for your AutoML model training. AutoML Vision models require GPU SKUs and support NC and ND families. Using a compute target with a multi-GPU VM SKU will leverage the multiple GPUs to speed up training. Additionally, setting up a compute target with multiple nodes will allow for faster model training by leveraging parallelism, when tuning hyperparameters for your model.

The compute target is a required parameter and is passed in using the `compute_target` parameter of the AutoMLVisionConfig. For example: 

```python
from azureml.train.automl import AutoMLVisionConfig
automl_vision_config = AutoMLVisionConfig(compute_target=compute_target)
```

### Configure model algorithms and hyperparameters
When using AutoML Vision to build vision models, users can control the model algorithm and sweep hyperparameters. These model algorithms and hyperparameters are passed in as the parameter space for the sweep. 

The model algorithm is required and is passed in via `model_name` parameter. You can either specify a single model_name or choose between multiple.

#### Currently supported model algorithms:
<ul>
<li><b>Image Classification (multi-class and multi-label):</b> 'resnet18', 'resnet34', 'resnet50', 'mobilenetv2', 'seresnext'</li>
<li><b>Object Detection: </b>'yolov5', 'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet34_fpn', 'fasterrcnn_resnet18_fpn', 'retinanet_resnet50_fpn'</li>
<li><b>Instance segmentation: </b>'maskrcnn_resnet50_fpn'</li>
</ul>

#### Hyperparameters for model training

In addition to controlling the model algorthm used, you can also tune hyperparameters used for model training. While many of the hyperparameters exposed are model-agnostic, some are task-specific and a few are model-specific.

The following tables list out the details of the hyperparameters  and their default values for each -

<b>Model-agnostic hyperparameters</b> 

| Parameter Name       | Description           | Default  |
| ------------- |-------------| -----|
| number_of_epochs | Number of training epochs <br> `Optional, Positive Integer` |  1. all (except yolov5) : 15 <br> 2. yolov5: 30 |
| training_batch_size | Training batch size <br> `Optional, Positive Integer` | 1. image classification: 78 <br> 2. object detection (except yolov5), instance segmentation: 2 <br> 3. yolov5: 16 |
| validation_batch_size | Validation batch size <br> `Optional, Positive Integer` | 1. image classification: 78 <br> 2. object detection (except yolov5), instance segmentation: 2 <br> 3. yolov5: 16  |
| early_stopping | Enable early stopping logic during training <br> `Optional, one of {0: False, 1: True}`| 1 |
| early_stopping_patience | Min number of epochs/validation evaluations <br>with no primary metric improvement <br>before the run is stopped <br> `Optional, Positive Integer` | 5 |
| early_stopping_delay | Min number of epochs/validation evaluations <br>to wait before primary metric improvement <br>is tracked for early stopping <br> `Optional, Positive Integer` | 5 |
| learning_rate | Initial learning rate <br> `Optional, float in [0, 1]` | 1. multi-class classification: 0.01 <br> 2. multi-label classification: 0.035 <br> 3. object detection (except yolov5), instance segmentation: 0.05  <br> 4. yolov5: 0.01  |
| lr_scheduler | Type of learning rate scheduler <br> `Optional, one of {warmup_cosine, step}` | warmup_cosine |
| step_lr_gamma | Value of gamma for the learning rate scheduler<br>if it is of type step <br> `Optional, float in [0, 1]` | 0.5 |
| step_lr_step_size | Value of step_size for the learning rate scheduler<br>if it is of type step <br> `Optional, Positive Integer` | 5 |
| warmup_cosine_lr_cycles | Value of cosine cycle for the learning rate scheduler<br>if it is of type warmup_cosine <br> `Optional, float in [0, 1]` | 0.45 |
| warmup_cosine_lr_warmup_epochs | Value of warmup epochs for the learning rate scheduler<br>if it is of type warmup_cosine <br> `Optional, Positive Integer` | 2 |
| optimizer | Type of optimizer <br> `Optional, one of {sgd, adam, adamw}`  | sgd |
| momentum | Value of momentum for the optimizer<br>if it is of type sgd <br> `Optional, float in [0, 1]` | 0.9 |
| weight_decay | Value of weight_decay for the optimizer<br>if it is of type sgd or adam or adamw <br> `Optional, float in [0, 1]` | 1e-4 |
| nesterov | Enable nesterov for the optimizer<br>if it is of type sgd <br> `Optional, one of {0: False, 1: True}`| 1 |
| beta1 | Value of beta1 for the optimizer<br>if it is of type adam or adamw <br> `Optional, float in [0, 1]` | 0.9 |
| beta2 | Value of beta2 for the optimizer<br>if it is of type adam or adamw <br> `Optional, float in [0, 1]` | 0.999 |
| amsgrad | Enable amsgrad for the optimizer<br>if it is of type adam or adamw <br> `Optional, one of {0: False, 1: True}` | 0 |
| evaluation_frequency | Frequency to evaluate validation dataset<br>to get metric scores <br> `Optional, Positive Integer` | 1 |
| split_ratio | Validation split ratio when splitting train data<br>into random train and validation subsets<br>if validation data is not defined <br> `Optional, float in [0, 1]` | 0.2 |
| checkpoint_frequency | Frequency to store model checkpoints.<br>By default, we save checkpoint at the epoch<br>which has the best primary metric on validation <br> `Optional, Positive Integer no greater than number_of_epochs` | no default value <br> (checkpoint at epoch with best primary metric)  |
| layers_to_freeze | How many layers to freeze for your model.<br>Available layers for each model is following: <br> {'resnet': [('conv1.', 'bn1.'), 'layer1.', 'layer2.', 'layer3.', 'layer4.'], <br>'mobilenetv2': ['features.0.', 'features.1.', 'features.2.', 'features.3.', 'features.4.', 'features.5.', 'features.6.', 'features.7.', 'features.8.', 'features.9.', 'features.10.', 'features.11.', 'features.12.', 'features.13.', 'features.14.', 'features.15.', 'features.16.', 'features.17.', 'features.18.'], <br> 'seresnext': ['layer0.', 'layer1.', 'layer2.', 'layer3.', 'layer4.'], <br> 'yolov5_backbone': ['model.0.', 'model.1.', 'model.2.', 'model.3.', 'model.4.', 'model.5.', 'model.6.', 'model.7.', 'model.8.', 'model.9.'], <br> 'resnet_backbone': ['backbone.body.conv1.', 'backbone.body.layer1.', 'backbone.body.layer2.', 'backbone.body.layer3.', 'backbone.body.layer4.']}. <br>For instance, passing 2 as value for seresnext means<br>you want to freeze layer0 and layer1.<br>If this is not specified, we default to:<br>no frozen layer for resnet18/34/50,<br>mobilenetv2, seresnext and yolov5,<br>while the first two layers are frozen in resnet backbone<br>for fasterrcnn, maskrcnn and retinanet. <br> `Optional, Positive Integer no greater than number of available layers for each model` | no default value |

<br>
<b>Task-specific hyperparameters</b> 
<br>
<b> For Image Classifcation (Multi-class and Multi-label)  - </b>

| Parameter Name       | Description           | Default  |
| ------------- |-------------| -----|
| weighted_loss | 0 for no weighted loss<br>1 for weighted loss with sqrt(class_weights),<br>and 2 for weighted loss with class_weights | 0 |
| resize_size | Image size to which to resize before cropping for validation dataset | 256  |
| crop_size | Image crop size which is input to your neural network | 224 |

<br>
<b>For Object Detection and Instance Segmentation - </b>  

| Parameter Name       | Description           | Default  |
| ------------- |-------------| -----|
| validation_metric_type | metric computation method to use for validation metrics | voc |
| min_size | Minimum size of the image to be rescaled before feeding it to the backbone | 600 |
| max_size | Maximum size of the image to be rescaled before feeding it to the backbone | 1333 |
| box_score_thresh | During inference, only return proposals with a classification score<br> greater than box_score_thresh | 0.3 |
| box_nms_thresh | NMS threshold for the prediction head. Used during inference | 0.5 |
| box_detections_per_img | Maximum number of detections per image, for all classes | 100 |

<br>
<b>Model-specific hyperparameters</b> 
<br>

<b>For yolov5 - </b>
| Parameter Name       | Description           | Default  |
| ------------- |-------------| -----|
| img_size | image size for train and val | 640 |
| model_size | model size (small, medium, large, xlarge) | medium |
| multi_scale | Enable multi-scale image by varying image size by +/- 50% | 0 |
| box_score_thresh | During inference, only return proposals with a score<br> greater than box_score_thresh. The score is the multiplication of<br> the objectness score and classification probability | 0.001 |
| box_iou_thresh | IOU threshold used during inference in nms post processing | 0.5 |
<br>

### Sweeping hyperparameters for your model
When training vision models, model performance depends heavily on the hyperparameter values selected. Often times, you might want to tune the hyperparameters to get optimal performance.  
AutoML Vision allows you to sweep hyperparameters to find the optimal settings for your model. It leverages the hyperparameter tuning capabilities in Azure Machine Learning - you can learn more [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters).

#### Define the parameter search space
You can define the model algorithms and hyperparameters to sweep in the parameter space. See [Configure model algorithms and hyperparameters](#Configure-model-algorithms-and-hyperparameters) for the list of supported model algorithms and hyperparameters for each task type. Details on supported distributions for discrete and continuous hyperparameters can be found [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#define-the-search-space).

#### Sampling methods for the sweep
When sweeping hyperparameters, you need to specify the sampling method to use for sweeping over the defined parameter space. AutoML Vision supports the following sampling methods using the `hyperparameter_sampling` parameter -
<ul>
<li>Random Sampling</li>
<li>Grid Sampling</li>
<li>Bayesian Sampling</li>
</ul>

You can learn more about each of these sampling methods [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#sampling-the-hyperparameter-space).

#### Early termination policies
When using AutoML Vision to sweep hyperparameters for your vision models, you can automatically end poorly performing runs with an early termination policy. Early termination improves computational efficiency, saving compute resources that would have been otherwise spent on less promising configurations. AutoML Vision supports the following early termination policies using the `policy` parameter -
<ul>
<li>Bandit Policy</li>
<li>Median Stopping Policy</li>
<li>Truncation Selection Policy</li>
</ul>

If no termination policy is specified, all configurations are run to completion.  
You can learn more about configuring the early termination policy for your hyperparameter sweep [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#early-termination).

#### Resources for the sweep
You can control the resources spent on your hyperparameter sweep by specifying the `iterations` and the `max_concurrent_iterations` for the sweep.
<ul>  
<li>iterations (required when sweeping): Maximum number of configurations to sweep. Must be an integer between 1 and 1000. </li>
<li>max_concurrent_iterations: (optional) Maximum number of runs that can run concurrently. If not specified, all runs launch in parallel. If specified, must be an integer between 1 and 100.  (NOTE: The number of concurrent runs is gated on the resources available in the specified compute target. Ensure that the compute target has the available resources for the desired concurrency.)</li>
</ul>

### Optimization metric
You can specify the metric to be used for model optimization and hyperparameter tuning using the optional `primary_metric` parameter. Default values depend on the task type -
<ul>
<li>'accuracy' for image-classification</li>
<li>'IOU' for image-classification-multi-label</li>
<li>'mean_average_precision' for image-object-detection</li>
<li>'mean_average_precision' for image-instance-segmentation</li>
</ul>

### Experiment budget
You can optionally specify the maximum time budget for your AutoML Viison experiment using `experiment_timeout_hours` - the amount of time in hours before the experiment terminates. If none specified, default experiment timeout is 6 days.

<!---
### Early stopping
You can optionally enable early stopping for your AutoML Vision experiment using `enable_early_stopping` parameter. 
| Parameter Name       | Description           | Default  |
| ------------- |-------------| -----|
| early_stopping | Enable early stopping logic during training | 1 |
| early_stopping_patience | Minimum number of epochs/validation evaluations<br> with no primary metric score improvement before the run is stopped | 5 |
| early_stopping_delay | Minimum number of epochs/validation evaluations<br> to wait before primary metric score improvement is tracked for early stopping | 5 |
<br>
-->
### Arguments
You can pass fixed settings or parameters that don't change during the parameter space sweep as arguments. Arguments are passed in name-value pairs and the name must be prefixed by a double dash. For example:

```python
from azureml.train.automl import AutoMLVisionConfig
arguments = ["--early_stopping", 1, "--evaluation_frequency", 2]
automl_vision_config = AutoMLVisionConfig(arguments=arguments)
```

## Sample notebooks
Please refer to the following sample notebooks to see how you can use AutoML Vision with sample data in your scenario -

Object Detection - [AutoML Vision Object Detection Sample Notebook](./ObjectDetection/AutoMLVision_ObjectDetection_SampleNotebook.ipynb)</br>
Multi-Class Classification - [AutoML Vision Multi-Class Classification Sample Notebook](./MultiClass/AutoMLVision_MultiClass_SampleNotebook.ipynb)</br>
Multi-Label Classification - [AutoML Vision Multi-Label Classification Sample Notebook](./MultiLabel/AutoMLVision_MultiLabel_SampleNotebook.ipynb)</br>
Instance Segmentation - [AutoML Vision Instance Segmentation Sample Notebook](./InstanceSegmentation/AutoMLVision_InstanceSegmentation_SampleNotebook.ipynb)</br>
