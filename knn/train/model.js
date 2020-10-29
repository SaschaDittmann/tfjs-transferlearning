const tf = require('@tensorflow/tfjs')
const knnClassifier = require('@tensorflow-models/knn-classifier')

class FlowersModel {
  constructor() {
    this.classifier = knnClassifier.create();
  }

  train (mobilenet, trainImages, trainLabels) {
    console.log('Starting classifier training...');
    for (let j = 0; j < trainImages.shape[0]; j++){
      const activation = mobilenet.predict(trainImages.slice(j, 1));
      this.classifier.addExample(activation, trainLabels[j]);
    }
    console.log('Classifier training completed.');
  }

  async predict (mobilenet, image){
    const activation = mobilenet.predict(image);
    return await this.classifier.predictClass(activation);
  }

  async evaluate(mobilenet, testImages, testLabels){
    console.log('Starting classifier evaluation...');
    var numOfCorrectPredictions = 0;
    for (let j = 0; j < testImages.shape[0]; j++){
        const result = await this.predict(mobilenet, testImages.slice(j, 1));

        if (result.label == testLabels[j]){
            numOfCorrectPredictions++;
        }
    }
    console.log('Classifier evaluation completed.');

    const accuracy = numOfCorrectPredictions / testImages.shape[0];
    return accuracy;
  }
}

module.exports = new FlowersModel();
