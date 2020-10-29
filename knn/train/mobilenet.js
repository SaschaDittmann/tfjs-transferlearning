const tf = require('@tensorflow/tfjs')
const MOBILENET_MODEL_PATH = 
  'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.75_224/model.json';

class MobilenetLoader {
  constructor() {
    this.mobilenetModel = null;
  }

  async loadModel() {
    console.log('Loading MobileNet model...');
    this.mobilenetModel = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
    console.log('Model loaded successfully.');
  }

  getModel() {
    return this.mobilenetModel;
  }
}

module.exports = new MobilenetLoader();
