# AutoML Vision

## AutoML Vision Overview

### What is AutoML Vision?
AutoML is an [Azure Machine Learning](https://azure.microsoft.com/en-us/services/machine-learning/) feature, that empowers both professional and citizen data scientists to build machine learning models rapidly. Since its launch, AutoML has helped accelerate model building for essential machine learning tasks like Classification, Regression and Time-series Forecasting. With the preview of AutoML Vision, there will be added support for Vision tasks. Data scientists of all skill levels will be able to easily generate models trained on image data for scenarios like Image Classification, Object Detection and Instance Segmentation. 

Customers across various industries are looking to leverage machine learning to build models that can process image data. Applications range from image classification of fashion photos to PPE detection in industrial environments. Customers want a solution to easily build models, controlling the model training to generate the optimal model for their training data, and a way to easily manage these ML models end-to-end. While Azure Machine Learning offers a solution for managing the end-to-end ML Lifecycle, customers currently have to rely on the tedious process of custom training their image models. Iteratively finding the right set of model algorithms and hyperparameters for these scenarios typically require significant data scientist effort.

With AutoML support for Vision tasks, Azure ML customers can easily build models trained on image data, without writing any training code. Customers can seamlessly integrate with Azure ML's Data Labeling capability and use this labeled data for generating image models. They can control the model generated by specifying the model algorithm and can optionally tune the hyperparameters. They can sweep over multiple model algorithms / hyperparameter ranges and find the optimal model for their needs. The resulting model can then be downloaded or deployed as a web service in Azure ML and can be operationalized at scale, leveraging AzureML MLOps capabilities.

Authoring AutoML models for vision tasks will be initially supported via the Azure ML Python SDK. UI authoring will be added in a subsequent release. The resulting experimentation runs, models and outputs will be accessible from the Azure ML Studio UI.

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

## How to use AutoML Vision building models for Vision tasks?