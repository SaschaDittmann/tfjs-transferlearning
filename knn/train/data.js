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
        .resizeNearestNeighbor([224,224])
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
      trainLabels: this.data[1],
      testImages: tf.concat(this.data[2]),
      testLabels: this.data[3],
      labelMappings: this.data[4]
    }
  }
}

module.exports = new FlowerDataset();
