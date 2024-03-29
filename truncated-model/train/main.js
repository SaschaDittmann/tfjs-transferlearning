const tf = require('@tensorflow/tfjs-node-gpu');

const data = require('./data');
const mobilenet = require('./mobilenet');
const model = require('./model');

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

    await mobilenet.loadModel();
    mobilenet.getModel().summary();

    const truncatedMobileNetModel = mobilenet.getTruncatedModel();
    truncatedMobileNetModel.summary();

    const flowersModel = model.getFlowersModel(truncatedMobileNetModel);
    flowersModel.summary();

    const batchSize = Math.floor(trainImages.shape[0] * batchSizeFraction);

    console.log("Preprocessing training images...");
    const trainImg = [] 
    for (let j = 0; j < trainImages.shape[0]; j++){
        embedding = truncatedMobileNetModel.predict(trainImages.slice(j, 1));
        trainImg.push(embedding);
    }
    var preprocessedTrainImages = tf.concat(trainImg);
    
    console.log("Preprocessing testing images...");
    const testImg = [] 
    for (let j = 0; j < testImages.shape[0]; j++){
        embedding = truncatedMobileNetModel.predict(testImages.slice(j, 1));
        testImg.push(embedding);
    }
    var preprocessedTestImages = tf.concat(testImg);
    
    console.log("Start training...");
    console.log("Epochs: " + epochs);
    console.log("Batch Size: " + batchSize);

    const validationSplit = 0.1;
    await flowersModel.fit(preprocessedTrainImages, trainLabels, {
        epochs,
        batchSize,
        validationSplit
    });

    const evalOutput = flowersModel.evaluate(preprocessedTestImages, testLabels);
    console.log(
        `\nEvaluation result:\n` +
        `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
        `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);

    if (modelSavePath != null) {
        await truncatedMobileNetModel.save(`file://${modelSavePath}/base`);
        await flowersModel.save(`file://${modelSavePath}/head`);
        console.log(`Saved models to path: ${modelSavePath}`);
    }
}

run(25, 0.4, './model');
