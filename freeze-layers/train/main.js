const tf = require('@tensorflow/tfjs-node-gpu');
const fs = require('fs');

const data = require('./data');
const modelBuilder = require('./model');

async function run(epochs, batchSizeFraction, modelSavePath) {
    data.loadData();

    const {
        images: trainImagesBase, 
        labels: trainLabelsBase, 
        mappings: labelMappingsBase,
    } = data.getBaseTrainData();
    numOfClassesBase = labelMappingsBase.length;
    console.log(
        `\nData for the base model:\n` +
        `Training Images: ${trainImagesBase.shape}\n`+
        `Training Labels: ${trainLabelsBase.shape}\n`+
        `Label Mappings : ${labelMappingsBase}`);
    
    let model;
    if (!fs.existsSync(`${modelSavePath}/base/model.json`)) {
        model = modelBuilder.getFlowersModel(numOfClassesBase.length);
        model.summary();

        console.log( "Training base model..." );
        const batchSize = Math.floor(trainImagesBase.shape[0] * batchSizeFraction);
        const validationSplit = 0.15;
        await model.fit(trainImagesBase, trainLabelsBase, {
            epochs,
            batchSize,
            validationSplit
        });
        console.log( "Base model training completed." );

        const {images: testImagesBase, labels: testLabelsBase} = data.getBaseTestData();
        const evalOutput = model.evaluate(testImagesBase, testLabelsBase);
        console.log(
            `\nEvaluation result for the base model:\n` +
            `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
            `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);

        console.log( "Saving base model to disk..." );
        await model.save(`file://${modelSavePath}/base`);
        console.log(`Saved model to path: ${modelSavePath}/base`);
    } else {
        console.log( "Loading base model from disk..." );
        model = await tf.loadLayersModel(`file://${modelSavePath}/base/model.json`);
        console.log( "Model loaded." );
    }
    
    const {
        images: trainImagesRetrain, 
        labels: trainLabelsRetrain, 
        mappings: labelMappingsRetrain,
    } = data.getRetrainTrainData();
    numOfClassesRetrain = labelMappingsBase.length + labelMappingsRetrain.length;
    console.log(
        `\nData for the retrain model:\n` +
        `Training Images: ${trainImagesRetrain.shape}\n`+
        `Training Labels: ${trainLabelsRetrain.shape}\n`+
        `Label Mappings : ${labelMappingsRetrain}`);

    // Freeze layers
    for (let i = 0; i < 15; ++i) {
        model.layers[i].trainable = false;
    }

    console.log( "Retraining model..." );
    const batchSizeRetrain = Math.floor(trainImagesRetrain.shape[0] * batchSizeFraction);
    const validationSplitRetrain = 0.15;
    model.compile({
        optimizer: tf.train.adam(0.0001),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy'],
    });
    await model.fit(trainImagesRetrain, trainLabelsRetrain, {
        epochs,
        batchSizeRetrain,
        validationSplitRetrain,
        callbacks: tf.callbacks.earlyStopping({monitor: 'acc'})
    });
    console.log( "Model retraining completed." );

    const {images: testImagesRetrain, labels: testLabelsRetrain} = data.getRetrainTestData();
    const evalOutput = model.evaluate(testImagesRetrain, testLabelsRetrain);
    console.log(
        `\nEvaluation result for the base model:\n` +
        `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
        `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);

    console.log( "Saving base model to disk..." );
    await model.save(`file://${modelSavePath}/retrain`);
    console.log(`Saved model to path: ${modelSavePath}/retrain`);
}

run(100, 0.1, './model');
