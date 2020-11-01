const tf = require('@tensorflow/tfjs')

class FlowersModel {
  constructor() {
    this.model = null;
  }

  getFlowersModel(numOfClasses) {
    if(!this.model){
        const kernel_size = [3, 3]
        const pool_size= [2, 2]
        const first_filters = 32
        const second_filters = 64
        const third_filters = 128
        const dropout_conv = 0.3
        const dropout_dense = 0.3

        this.model = tf.sequential();
        this.model.add(tf.layers.conv2d({
            inputShape: [96, 96, 3],
            filters: first_filters,
            kernelSize: kernel_size,
            activation: 'relu',
            }));
        this.model.add(tf.layers.conv2d({
            filters: first_filters,
            kernelSize: kernel_size,
            activation: 'relu',
            }));
        this.model.add(tf.layers.maxPooling2d({poolSize: pool_size}));
        this.model.add(tf.layers.dropout({rate: dropout_conv}));

        this.model.add(tf.layers.conv2d({
            filters: second_filters,
            kernelSize: kernel_size,
            activation: 'relu',
            }));
        this.model.add(tf.layers.conv2d({
            filters: second_filters,
            kernelSize: kernel_size,
            activation: 'relu',
            }));
        this.model.add(tf.layers.conv2d({
            filters: second_filters,
            kernelSize: kernel_size,
            activation: 'relu',
            }));
        this.model.add(tf.layers.maxPooling2d({poolSize: pool_size}));
        this.model.add(tf.layers.dropout({rate: dropout_conv}));

        this.model.add(tf.layers.conv2d({
            filters: third_filters,
            kernelSize: kernel_size,
            activation: 'relu',
            }));
        this.model.add(tf.layers.conv2d({
            filters: third_filters,
            kernelSize: kernel_size,
            activation: 'relu',
            }));
        this.model.add(tf.layers.conv2d({
            filters: third_filters,
            kernelSize: kernel_size,
            activation: 'relu',
            }));
        this.model.add(tf.layers.maxPooling2d({poolSize: pool_size}));
        this.model.add(tf.layers.dropout({rate: dropout_conv}));

        this.model.add(tf.layers.flatten());

        this.model.add(tf.layers.dense({units: 256, activation: 'relu'}));
        this.model.add(tf.layers.dropout({rate: dropout_dense}));
        this.model.add(tf.layers.dense({units: numOfClasses, activation: 'softmax'}));

        this.model.compile({
            optimizer: tf.train.adam(0.0001),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy'],
        });
    }

    return this.model;
  }
}

module.exports = new FlowersModel();
