const tf = require('@tensorflow/tfjs')
const MOBILENET_MODEL_PATH = 
  'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';

class MobilenetLoader {
  constructor() {
    this.model = null;
  }

  async loadModel() {
    console.log('Loading model...');
    this.model = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
    console.log('Model loaded successfully.');
  }

  getModel() {
    return this.model;
  }
}

module.exports = new MobilenetLoader();
