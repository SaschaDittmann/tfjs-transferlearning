const tf = require('@tensorflow/tfjs-node-gpu');

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
    
    const modelBase = modelBuilder.getFlowersModel(labelMappingsBase.length);
    modelBase.summary();

    const batchSize = Math.floor(trainImagesBase.shape[0] * batchSizeFraction);
    const validationSplit = 0.15;
    await modelBase.fit(trainImagesBase, trainLabelsBase, {
        epochs,
        batchSize,
        validationSplit
    });

    const {images: testImagesBase, labels: testLabelsBase} = data.getBaseTestData();
    const evalOutput = modelBase.evaluate(testImagesBase, testLabelsBase);
    console.log(
        `\nEvaluation result for the base model:\n` +
        `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
        `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);

    if (modelSavePath != null) {
        await modelBase.save(`file://${modelSavePath}/base`);
        console.log(`Saved model to path: ${modelSavePath}/base`);
    }
}

run(100, 0.1, './model');
