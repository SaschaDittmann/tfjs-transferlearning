const tf = require('@tensorflow/tfjs')

class MobilenetLoader {
  constructor() {
    this.model = null;
    this.mobilenetModel = null;
  }

  async loadModel() {
    console.log('Loading MobileNetV2 model...');
    this.mobilenetModel = await tf.loadLayersModel('file://mobilenetv2/model.json', {strict: false});
    console.log('Model loaded successfully.');
  }

  getModel() {
    return this.mobilenetModel;
  }

  getTruncatedModel() {
    if(!this.model){
      var layer = this.mobilenetModel.getLayer('out_relu');
      this.model = tf.model({
        inputs: this.mobilenetModel.inputs, 
        outputs: layer.output
      });
    }

    return this.model;
  }
}

module.exports = new MobilenetLoader();
