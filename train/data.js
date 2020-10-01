const tf = require('@tensorflow/tfjs-node-gpu');
const fs = require('fs');
const path = require('path');

const IMAGES_DIR = './data';

function loadImages(dataDir) {
  const trainImages = [];
  const trainLabels = [];
  const testImages = [];
  const testLabels = [];
  const labelMappings = [];

  dirs = fs.readdirSync(dataDir);
  for (let i = 0; i < dirs.length; i++) { 
    var dirPath = path.join(dataDir, dirs[i]);
    if (!fs.statSync(dirPath).isDirectory()) {
      continue;
    }
    labelMappings.push(dirs[i]);
    
    var files = fs.readdirSync(dirPath);
    for (let j = 0; j < files.length; j++) { 
      if (!files[j].toLocaleLowerCase().endsWith(".jpg")) {
        continue;
      }

      var filePath = path.join(dirPath, files[j]);
      // console.log(filePath);

      var buffer = fs.readFileSync(filePath);
      var imageTensor = tf.node.decodeImage(buffer)
        .resizeNearestNeighbor([224,224])
        .toFloat()
        .div(tf.scalar(255.0))
        .expandDims();

      if (j % 5 == 0){
        testImages.push(imageTensor);
        testLabels.push(i);
      } else {
        trainImages.push(imageTensor);
        trainLabels.push(i);
      }
    }
  }
  
  return [trainImages, trainLabels, testImages, testLabels, labelMappings];
}

/** Helper class to handle loading training and test data. */
class FlowerDataset {
  constructor() {
    this.data = [];
  }

  /** Loads training and test data. */
  loadData() {
    console.log('Loading images...');
    this.data = loadImages(IMAGES_DIR);
    console.log('Images loaded successfully.')
  }

  getData() {
    return {
      trainImages: tf.concat(this.data[0]),
      trainLabels: tf.oneHot(tf.tensor1d(this.data[1], 'int32'), this.data[4].length).toFloat(),
      testImages: tf.concat(this.data[2]),
      testLabels: tf.oneHot(tf.tensor1d(this.data[3], 'int32'), this.data[4].length).toFloat(),
      labelMappings: this.data[4]
    }
  }
}

module.exports = new FlowerDataset();
