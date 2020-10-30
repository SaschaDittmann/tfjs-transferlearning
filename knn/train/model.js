const tf = require('@tensorflow/tfjs')
const knnClassifier = require('@tensorflow-models/knn-classifier')
const fs = require('fs');

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

  async toDatasetObject(dataset) {
    const datasetObj = await Promise.all(
      Object.entries(dataset).map(async ([classId,value], index) => {
        const data = await value.data();
  
        return {
          classId: Number(classId),
          data: Array.from(data),
          shape: value.shape
        };
     })
    );

    return datasetObj;
  }

  fromDatasetObject(datasetObject) {
    return Object.entries(datasetObject).reduce((result, [indexString, {data, shape}]) => {
      const tensor = tf.tensor2d(data, shape);
      const index = Number(indexString);
  
      result[index] = tensor;
  
      return result;
    }, {});
  }

  async save(filename) {
    console.log(`Saving classifier to file ${filename}...`);

    let dataset = this.classifier.getClassifierDataset()
    const datasetObj = await this.toDatasetObject(dataset);
    let datasetJson = JSON.stringify(datasetObj)
    
    fs.writeFile(filename, datasetJson, 'utf8', function (err) {
      if (err) throw err;
      console.log('Classifier saved successfully.');
    });
  }

  async load(filename) {
    console.log(`Loading classifier from file ${filename}...`);
    const datasetJson = fs.readFileSync(filename, 'utf8');
    let datasetObj = JSON.parse(datasetJson);
    const dataset = this.fromDatasetObject(datasetObj);

    const classifier = new knnClassifier.KNNClassifier();
    classifier.setClassifierDataset(dataset);

    this.classifier.dispose(); // clear old classifier 
    this.classifier = classifier;
    console.log('Classifier loaded successfully.');
  }
}

module.exports = new FlowersModel();
