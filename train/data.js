const tf = require('@tensorflow/tfjs-node-gpu');
const fs = require('fs');
const path = require('path');

const IMAGES_DIR = './data';

function loadImages(dataDir) {
  const images = [];
  const labels = [];
  var labelCount = 0;

  dirs = fs.readdirSync(dataDir);
  for (let i = 0; i < dirs.length; i++) { 
    var dirPath = path.join(dataDir, dirs[i]);
    if (!fs.statSync(dirPath).isDirectory()) {
      continue;
    }
    labelCount++;
    
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

      images.push(imageTensor);
      labels.push(dirs[i]);
    }
  }
  
  return [images, labels, labelCount];
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
      images: tf.concat(this.data[0]),
      labels: tf.oneHot(tf.tensor1d(this.data[1], 'int32'), this.data[2]).toFloat(),
      numOfClasses: this.data[2]
    }
  }
}

module.exports = new FlowerDataset();
