# AutoML Vision

## AutoML Vision Overview

### What is AutoML Vision?
AutoML is an [Azure Machine Learning](https://azure.microsoft.com/en-us/services/machine-learning/) feature, that empowers both professional and citizen data scientists to build machine learning models rapidly. Since its launch, AutoML has helped accelerate model building for essential machine learning tasks like Classification, Regression and Time-series Forecasting. With the preview of AutoML Vision, there will be added support for Vision tasks. Data scientists will be able to easily generate models trained on image data for scenarios like Image Classification, Object Detection and Instance Segmentation. 

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
<li>image-classification-multi-label</li>
<li>image-object-detection</li>
<li>image-instance-segmentation</li>
</ul>

This task type is a required parameter and is passed in using the `task` parameter in the AutoMLVisionConfig For e.g. 

```python
from azureml.train.automl import AutoMLVisionConfig
automl_vision_config = AutoMLVisionConfig(task = 'image-object-detection')
```

### Training and Validation data
In order to generate Vision models, you will need to bring in labeled image data as input for model training in the form of an AzureML `Labeled Dataset`. You can either use a Labeled Dataset that you have exported from a Data Labeling project, or create a new Labeled Dataset with your labeled training data. 

Labeled datasets are [Tabular datasets](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.tabulardataset)  with some enhanced capabilites such as mounting and downloading. Structure of the labeled dataset depends upon the task at hand. For Image task types, it consists of the following fields:
<ul>
<li>image_url: contains filepath as a StreamInfo object</li>
<li>image_details: image metatadata information consist of height, width and format. This field is optional and hence may or may not exist. This restriction might become mandatory in the future.</li>
<li>label: a json representation of the image label, based on the task type</li>
</ul>

Creation of labeled datasets is supported from data in JSONL format. If your training data is in a different format (e.g. pascal VOC), you can leverage the helper scripts included with the sample notebooks in this repo to convert the data to JSONL. Once your data is in JSONL format, you can create a labeled dataset using this snippet -

```python
from azureml.contrib.dataset.labeled_dataset import _LabeledDatasetFactory, LabeledDatasetTask
from azureml.core import Dataset

training_dataset = _LabeledDatasetFactory.from_json_lines(
        task=LabeledDatasetTask.OBJECT_DETECTION, path=ds.path('odFridgeObjects/odFridgeObjects.jsonl'))
    training_dataset = training_dataset.register(workspace=ws, name=training_dataset_name)
```

You can optionally specify another labeled dataset as a validation dataset to be used for your model. If no validation dataset is specified, 20% of your training data will be used for validation.

Training data is a required parameter and is passed in using the `training_data` parameter. Validation data is optional and is passed in using the `validation_data` parameter of the AutoMLVisionConfig. For e.g. 

```python
from azureml.train.automl import AutoMLVisionConfig
automl_vision_config = AutoMLVisionConfig(training_data = training_dataset)
```

### Compute to run experiment
You will need to provide a [Compute Target](https://docs.microsoft.com/azure/machine-learning/service/concept-azure-machine-learning-architecture#compute-target) that will be used for your AutoML model training. AutoML Vision models require GPU SKUs and support NC and ND families. Using a compute target with a multi-GPU VM SKU will leverage the multiple GPUs to speed up training. Additionally, setting up a compute target with multiple nodes will allow for faster model training by leveraging parallelism, when tuning hyperparameters for your model.

The compute target is a required parameter and is passed in using the `compute_target` parameter of the AutoMLVisionConfig. For e.g. 

```python
from azureml.train.automl import AutoMLVisionConfig
automl_vision_config = AutoMLVisionConfig(compute_target = compute_target)
```

### Configure model algorithms and hyperparameters
When using AutoML Vision to build vision models, users can control the model algorithm and sweep hyperparameters. These model algorithms and hyperparameters are passed in as the parameter space for the sweep. 

The model algorithm is required and is passed in via `model_name` parameter. You can either specify a single model_name or choose between multiple.

#### Currently supported model algorithms:
<ul>
<li><b>Image Classification (multi-class and multi-label):</b> 'resnet18', 'resnet50', 'mobilenetv2', 'seresnext'</li>
<li><b>Object Detection: </b>'yolov5', 'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet34_fpn', 'fasterrcnn_resnet18_fpn', 'retinanet_resnet50_fpn'</li>
<li><b>Instance segmentation: </b>'maskrcnn_resnet50_fpn'</li>
</ul>

#### Hyperparameters for model training

In addition to controlling the model algorthm used, you can also tune hyperparameters used for model training. The hyperparameters exposed, and theur default values depend on the task type and the algorithm selected. The following tables list out the details of the hyperparameters for each -

<b> For Image Classifcation (Multi-class)</b>

| Parameter Name       | Description           | Default  |
| ------------- |-------------| -----|
| learning_rate | Initial learning rate |  0.01 |
| number_of_epochs | Number of training epochs |  15 |
| training_batch_size | Training batch size |  78 |
| validation_batch_size | Validation batch size |  78 |
| early_stopping | Enable early stopping logic during training |  True |
| early_stopping_patience | Minimum number of epochs/validation evaluations with no primary metric score improvement before the run is stopped |  5 |
| early_stopping_delay | Minimum number of epochs/validation evaluations to wait before primary metric score improvement is tracked for early stopping |  5 |
| optimizer | Type of optimizer in {sgd, adam, adamw} | sgd |
| momentum | Value of momentum for the optimizer if it is of type sgd |  0.9 |
| weight_decay | Value of weight_decay for the optimizer if it is of type sgd or adam or adamw |  1e-4 |
| nesterov | Enable nesterov for the optimizer if it is of type sgd |  True |
| beta1 | Value of beta1 for the optimizer if it is of type adam or adamw |  0.9 |
| beta2 | Value of beta2 for the optimizer if it is of type adam or adamw |  0.999 |
| amsgrad | Enable amsgrad for the optimizer if it is of type adam or adamw |  False |
| lr_scheduler | Type of learning rate scheduler in {warmup_cosine, step} | warmup_cosine |
| step_lr_gamma | Value of gamma for the learning rate scheduler if it is of type step |  0.5 |
| step_lr_step_size | Value of step_size for the learning rate scheduler if it is of type step |  5 |
| warmup_cosine_lr_cycles | Value of cosine cycle for the learning rate scheduler if it is of type warmup_cosine |  0.45 |
| warmup_cosine_lr_warmup_epochs | Value of warmup epochs for the learning rate scheduler if it is of type warmup_cosine |  2 |
| evaluation_frequency | Frequency to evaluate validation dataset to get metric scores |  1 |
| detailed_metrics | Report detailed metrics like per class/sample f1, f2, precision, recall scores |  True |
| imbalance_rate_threshold | data imbalance ratio (#data from largest class /#data from smallest class) |  2 |
| test_ratio |  |  0.2 |
| weighted_loss | applying class-level weighting in weighted loss for class imbalance |  0 |




### Sweeping hyperparameters for your model



#### Sampling methods for the sweep
#### Early termination policies
#### Resources for the sweep
### Optimization metric
### Experiment budget
## Sample notebooks