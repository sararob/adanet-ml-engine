**This is not a Google product.**

Training AdaNet models on Cloud ML Engine
============================

This is the code accompanying [this blog post](https://medium.com/tensorflow/combining-multiple-tensorflow-hub-modules-into-one-ensemble-network-with-adanet-56fa73588bb0) showing how to build an [AdaNet](https://github.com/tensorflow/adanet) model with `AutoEnsembleEstimator` and [TF Hub](https://tensorflow.org/hub), and train it on [Cloud ML Engine](https://cloud.google.com/ml-engine/).

See the blog post for details, and follow along below for a quick guide on how to train on ML Engine.

## Prerequisits: Setting up your Cloud project

For this to work you'll need to [create an account and project](https://cloud.google.com/resource-manager/docs/creating-managing-projects) on Google Cloud Platform, and enable billing, and the necessary APIs. Steps 1-4 on [this quickstart](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction#setup) explain how to do that.

## Install gcloud CLI

You'll use `gcloud` to kick off and manage training jobs for your model. If you don't have it, install it [here](https://cloud.google.com/sdk/gcloud/).

## Create a Cloud Storage bucket

You'll use this to store all of the checkpoints for your model along with the final model export. Follow [this guide](https://cloud.google.com/storage/docs/creating-buckets) to create one.

## Running a training job on Cloud ML Engine

Time to start your training. Open your terminal and make sure `gcloud` is set to the project you created for this tutorial: `gcloud config set project your-project-name`.

Define the following environment variables:

```bash
export JOB_ID=unique_job_name
export JOB_DIR=gs://your/gcs/bucket/path
export PACKAGE_PATH=trainer/
export MODULE=trainer.model
export REGION=your_cloud_project_region
```

From the root directory of this repo, run the following command:

```bash
gcloud ml-engine jobs submit training $JOB_ID --package-path trainer/ --module-name trainer.author --job-dir $JOB_DIR --region $REGION --runtime-version "1.12" --python-version 3.5 --config config.yaml
```

Navigate to the ML Engine UI in your cloud console to monitor the progress of your job. 

### Visualizing training progress with TensorBoard

You can also visualize metrics for your training job with [TensorBoard](). If you've got TensorFlow installed locally, it already comes with TensorBoard. Run the following command to start up TensorBoard:

```bash
tensorboard --logdir=$JOB_DIR
```

To start it, navigate to `localhost:6006` in your browser.

### Making a local prediction on your trained model

Once you've trained your model, it'll export the latest checkpoint to the Cloud Storage bucket path you specified. To quickly test out your model for prediction, you can use the `local predict` method via `gcloud`. Just create a *newline delimited* JSON file with your test instances in the format your model is expecting. An example file for this model is in `trainer/test-instances.json`. Then run:

```bash
gcloud ml-engine local predict --model-dir=gs://path/to/saved_model.pb --json-instances=path/to/test.json
```
