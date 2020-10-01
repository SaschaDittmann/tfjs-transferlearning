const tf = require('@tensorflow/tfjs-node-gpu');

const data = require('./data');
const modelLoader = require('./model');

async function run(epochs, batchSizeFraction, modelSavePath) {
    data.loadData();

    const {
        trainImages: trainImages, 
        trainLabels: trainLabels, 
        testImages: testImages, 
        testLabels: testLabels,      
        labelMappings: labelMappings
    } = data.getData();
    numOfClasses = labelMappings.length;
    console.log("Training Images (Shape): " + trainImages.shape);
    console.log("Training Labels (Shape): " + trainLabels.shape);
    console.log("Test Images (Shape): " + testImages.shape);
    console.log("Test Labels (Shape): " + testLabels.shape);
    console.log("Labels: " + labelMappings);

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
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    const batchSize = Math.floor(trainImages.shape[0] * batchSizeFraction);

    console.log("Preprocessing images...");
    var preprocessedTrainImages = truncatedMobileNetModel.predict(trainImages);
    var preprocessedTestImages = truncatedMobileNetModel.predict(testImages);

    console.log("Start training...");
    console.log("Epochs: " + epochs);
    console.log("Batch Size: " + batchSize);

    const validationSplit = 0.1;
    await model.fit(preprocessedTrainImages, trainLabels, {
        epochs,
        batchSize,
        validationSplit
    });

    const evalOutput = model.evaluate(preprocessedTestImages, testLabels);
    console.log(
        `\nEvaluation result:\n` +
        `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
        `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);

    if (modelSavePath != null) {
        await truncatedMobileNetModel.save(`file://${modelSavePath}/pre`);
        await model.save(`file://${modelSavePath}/main`);
        console.log(`Saved models to path: ${modelSavePath}`);
    }
    /*
    //**** IDEAS FROM THE ORIGINAL PROJECT ****

    model.compile({
        loss: 'categoricalCrossentropy',
        optimizer: tf.train.sgd(0.005),
        metrics: ['accuracy'],
    });
    */
}

run(20, 0.4, './model');
