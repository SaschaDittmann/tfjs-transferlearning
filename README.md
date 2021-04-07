[![Software License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![](https://ga4gh.azurewebsites.net/api?repo=tfjs-transferlearning)](https://github.com/SaschaDittmann/gaforgithub)

# TensorFlow.JS - Transfer Learning Examples

These examples show different approaches to retrain a pre-trained CNN model ([MobileNet v1](https://github.com/tensorflow/tfjs-models/tree/master/mobilenet)) with new images.

The [Flowers images](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz) used in these examples, are from the TensorFlow website and are not published with this repository.

## Setup 

For each subfolder (*train* or *web*) in each example

Prepare the node environments:
```sh
$ npm install
# Or
$ yarn
```

(Re)train the models (*train* subfolders) by running:
```sh
$ npm run start
# Or
$ yarn run start
```

Run the web server script (*web* subfolders) for testing the trained models in your local browser:
```sh
$ npm run start
# Or
$ yarn run start
```

## Demo

If you wan't, you can test the deployed application under [https://tfjs-transferlearning.azureedge.net/](https://tfjs-transferlearning.azureedge.net/).
