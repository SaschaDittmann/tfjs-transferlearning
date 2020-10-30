$("#image-selector").change(function () {
	let reader = new FileReader();
	reader.onload = function () {
		let dataURL = reader.result;
		$("#selected-image").attr("src", dataURL);
		$("#prediction-list").empty();
	}
	
	let file = $("#image-selector").prop('files')[0];
	reader.readAsDataURL(file);
});

function fromDatasetObject(datasetObject) {
	return Object.entries(datasetObject).reduce((result, [indexString, {data, shape}]) => {
		const tensor = tf.tensor2d(data, shape);
		const index = Number(indexString);

		result[index] = tensor;

		return result;
	}, {});
}

const MOBILENET_MODEL_PATH = 
  'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.75_224/model.json';
let mobilenetModel;
const classifier = knnClassifier.create();
$( document ).ready(async function () {
	$('.progress-bar').show();
	console.log( "Loading MobileNet model..." );
	mobilenetModel = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
	console.log( "MobileNet model loaded." );

	console.log( "Loading KNN model..." );
	var request = new XMLHttpRequest();
	request.onload = function() {
        if (request.readyState === 4) {
            if (request.status === 200) {
				let datasetObj = JSON.parse(request.responseText);
    			const dataset = fromDatasetObject(datasetObj);
				classifier.setClassifierDataset(dataset);
				console.log( "KNN model loaded." );
				$('.progress-bar').hide();
            } else {
                console.error(request.statusText);
            }
        }
	};
	request.onerror = function (e) {
		console.error(request.statusText);
	};
	request.open('GET', '/model/model.json');
	request.send(null);
});

$("#predict-button").click(async function () {
	let image = $('#selected-image').get(0);
	
	// Pre-process the image
	let tensor = tf.browser.fromPixels(image)
		.resizeNearestNeighbor([224,224])
		.toFloat()
		.div(tf.scalar(255.0))
		.expandDims();

	const activation = await mobilenetModel.predict(tensor);
	const prediction = await classifier.predictClass(activation);
	
	let top5 = Object.values(prediction.confidences)
		.map(function (p, i) { 
			return {
				probability: p,
				className: TARGET_CLASSES[i]
			};
		}).sort(function (a, b) {
			return b.probability - a.probability;
		}).slice(0, 5);

	$("#prediction-list").empty();
	top5.forEach(function (p) {
		$("#prediction-list").append(`<li>${p.className}: ${p.probability.toFixed(6)}</li>`);
		});
});
