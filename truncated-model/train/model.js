const tf = require('@tensorflow/tfjs')

class FlowersModel {
  constructor() {
    this.model = null;
  }

  getFlowersModel(truncatedModel) {
    if(!this.model){
        this.model = tf.sequential({
            layers: [
                tf.layers.flatten({
                    inputShape: truncatedModel.outputs[0].shape.slice(1)
                }),
                tf.layers.dense({
                    units: 256,
                    activation: 'relu',
                    kernelInitializer: 'varianceScaling',
                    useBias: true
                }),
                tf.layers.dropout({
                    rate: 0.2
                }),
                tf.layers.dense({
                    units: numOfClasses,
                    activation: 'softmax',
                    kernelInitializer: 'varianceScaling',
                    kernelRegularizer: tf.regularizers.l2({l2: 0.0001}),
                    useBias: true
                })
            ]
        });

        this.model.compile({
            optimizer: tf.train.adam(0.0001), 
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
    }

    return this.model;
  }
}

module.exports = new FlowersModel();
