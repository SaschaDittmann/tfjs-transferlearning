const tf = require('@tensorflow/tfjs-node-gpu');

const data = require('./data');
const modelLoader = require('./model');

async function run(epochs, batchSizeFraction, modelSavePath) {
    data.loadData();

    const {
        images: trainImages, 
        labels: trainLabels, 
        numOfClasses: numOfClasses
    } = data.getData();
    console.log("Training Images (Shape): " + trainImages.shape);
    console.log("Training Labels (Shape): " + trainLabels.shape);
    
    await modelLoader.loadModel();
    const truncatedMobileNetModel = modelLoader.getTruncatedMobileNetModel();
    //truncatedMobileNetModel.summary();

    // create new model
    const model = tf.sequential({
        layers: [
            tf.layers.flatten({
                inputShape: truncatedMobileNetModel.outputs[0].shape.slice(1)
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
    //model.summary();
   
    model.compile({
        optimizer: tf.train.adam(0.0001), 
        loss: 'categoricalCrossentropy'
    });

    const batchSize =
        Math.floor(trainImages.shape[0] * batchSizeFraction);

    //TODO: train model

    /*
    model.compile({
        loss: 'categoricalCrossentropy',
        optimizer: tf.train.sgd(0.005),
        metrics: ['accuracy'],
    });
    
    const validationSplit = 0.15;
    await model.fit(trainImages, trainLabels, {
        epochs,
        batchSize,
        validationSplit
    });

    if (modelSavePath != null) {
        await model.save(`file://${modelSavePath}`);
        console.log(`Saved model to path: ${modelSavePath}`);
    }
    */
}

run(20, 0.4, './model');
