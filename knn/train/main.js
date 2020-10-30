const data = require('./data');
const mobilenet = require('./mobilenet');
const model = require('./model');

async function run(modelSavePath) {
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
    console.log("Training Labels (Shape): " + trainLabels.length);
    console.log("Test Images (Shape): " + testImages.shape);
    console.log("Test Labels (Shape): " + testLabels.length);
    console.log("Labels: " + labelMappings);

    await mobilenet.loadModel();
    const net = mobilenet.getModel();
    net.summary();

    model.train(net, trainImages, trainLabels);

    const evalAccuracy = await model.evaluate(net, testImages, testLabels);
    console.log(
        `\nEvaluation result:\n` +
        `Accuracy = ${evalAccuracy.toFixed(3)}`);

    if (modelSavePath != null) {
        await model.save(modelSavePath);
    }
}

run('./model/model.json');
