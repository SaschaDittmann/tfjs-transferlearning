const tfnode = require('@tensorflow/tfjs-node-gpu');

const data = require('./data');
const modelLoader = require('./model');

async function run() {
    data.loadData();

    const {images: trainImages, labels: trainLabels} = data.getData();
    console.log("Training Images (Shape): " + trainImages.shape);
    console.log("Training Labels (Shape): " + trainLabels.shape);

    await modelLoader.loadModel();
    const model = modelLoader.getModel();
    console.log(model);

    // TODO: Train Model
}

run();
