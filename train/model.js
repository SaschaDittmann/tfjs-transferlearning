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
                    units: 100,
                    activation: 'relu',
                    kernelInitializer: 'varianceScaling',
                    useBias: true
                }),
                tf.layers.dense({
                    units: numOfClasses,
                    activation: 'softmax',
                    kernelInitializer: 'varianceScaling',
                    useBias: true
                })
            ]
        });

        this.model.compile({
            optimizer: tf.train.adam(0.0001), 
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        /*
        //**** IDEAS FROM THE ORIGINAL PROJECT ****

        model.compile({
            loss: 'categoricalCrossentropy',
            optimizer: tf.train.sgd(0.005),
            metrics: ['accuracy'],
        });
        */
    }

    return this.model;
  }
}

module.exports = new FlowersModel();
