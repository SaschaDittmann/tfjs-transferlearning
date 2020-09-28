const tf = require('@tensorflow/tfjs')
const mobilenet = require('@tensorflow-models/mobilenet');

class MobilenetLoader {
  constructor() {
    this.model = null;
  }

  async loadModel() {
    console.log('Loading model...');
    this.model = await mobilenet.load();
    console.log('Model loaded successfully.');
  }

  getModel() {
    return this.model;
  }
}

module.exports = new MobilenetLoader();
