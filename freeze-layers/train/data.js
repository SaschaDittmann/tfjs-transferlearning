const tf = require('@tensorflow/tfjs-node-gpu');
const fs = require('fs');
const path = require('path');

const IMAGES_DIR1 = './data1';
const IMAGES_DIR2 = './data2';

function loadImages(dataDir) {
  const trainImages = [];
  const trainLabels = [];
  const testImages = [];
  const testLabels = [];
  const labelMappings = [];

  dirs = fs.readdirSync(dataDir);
  classId = 0;
  for (let i = 0; i < dirs.length; i++) { 
    var dirPath = path.join(dataDir, dirs[i]);
    if (!fs.statSync(dirPath).isDirectory()) {
      continue;
    }
    labelMappings.push(dirs[i]);
    console.log("Scanning " + dirPath);
    
    var files = fs.readdirSync(dirPath);
    console.log("  found " + files.length + " files");
    for (let j = 0; j < files.length; j++) { 
      if (!files[j].toLocaleLowerCase().endsWith(".jpg")) {
        continue;
      }

      var filePath = path.join(dirPath, files[j]);
      // console.log(filePath);

      var buffer = fs.readFileSync(filePath);
      var imageTensor = tf.node.decodeImage(buffer)
        .resizeNearestNeighbor([96,96])
        .toFloat()
        .div(tf.scalar(255.0))
        .expandDims();

      if (j % 5 == 0){
        testImages.push(imageTensor);
        testLabels.push(classId);
      } else {
        trainImages.push(imageTensor);
        trainLabels.push(classId);
      }
    }
    classId++;
  }
  
  return [trainImages, trainLabels, testImages, testLabels, labelMappings];
}

/** Helper class to handle loading training and test data. */
class FlowerDataset {
  constructor() {
    this.baseData = [];
    this.retrainData = [];
  }

  /** Loads training and test data. */
  loadData() {
    console.log('Loading images...');
    this.baseData = loadImages(IMAGES_DIR1);
    this.retrainData = loadImages(IMAGES_DIR2);
    console.log('Images loaded successfully.')
  }

  getBaseTrainData() {
    return {
      images: tf.concat(this.baseData[0]),
      labels: tf.oneHot(tf.tensor1d(this.baseData[1], 'int32'), this.baseData[4].length).toFloat(),
      mappings: this.baseData[4]
    }
  }

  getRetrainTrainData() {
    const images = this.baseData[0].concat(this.retrainData[0]);
    const labels = this.baseData[1].concat(this.retrainData[1]);
    const mappings = this.baseData[4].concat(this.retrainData[4]);
    return {
      images: tf.concat(images),
      labels: tf.oneHot(tf.tensor1d(labels, 'int32'), mappings.length).toFloat(),
      mappings: mappings
    }
  }

  getBaseTestData() {
    return {
      images: tf.concat(this.baseData[2]),
      labels: tf.oneHot(tf.tensor1d(this.baseData[3], 'int32'), this.baseData[4].length).toFloat()
    }
  }

  getRetrainTestData() {
    const images = this.baseData[2].concat(this.retrainData[2]);
    const labels = this.baseData[3].concat(this.retrainData[3]);
    const mappings = this.baseData[4].concat(this.retrainData[4]);
    return {
      images: tf.concat(images),
      labels: tf.oneHot(tf.tensor1d(labels, 'int32'), mappings.length).toFloat()
    }
  }
}

module.exports = new FlowerDataset();
