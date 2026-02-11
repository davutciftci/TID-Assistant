const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

const DATA_PATH = path.join(process.cwd(), 'dataset_processed.json');
const MODEL_SAVE_PATH = path.join(process.cwd(), 'backend', 'model');

const TARGET_FRAMES = 30;
const FEATURES = 126;

async function train() {
    console.log('Loading dataset...');
    const rawData = JSON.parse(fs.readFileSync(DATA_PATH, 'utf8'));

    const labels = [...new Set(rawData.map((d) => d.label))];
    const labelMap = new Map(labels.map((l, i) => [l, i]));
    console.log(`Classes detected: ${labels.join(', ')}`);
    console.log(`Total samples: ${rawData.length}`);

    const X = [];
    const y = [];

    for (const sample of rawData) {
        const keypoints = sample.keypoints;
        const frames = sample.frames;

        let processedKeypoints;
        if (frames >= TARGET_FRAMES) {
            processedKeypoints = keypoints.slice(0, TARGET_FRAMES * FEATURES);
        } else {
            processedKeypoints = [...keypoints];
            const padding = new Array((TARGET_FRAMES - frames) * FEATURES).fill(0);
            processedKeypoints.push(...padding);
        }

        const reshaped = [];
        for (let i = 0; i < TARGET_FRAMES; i++) {
            reshaped.push(processedKeypoints.slice(i * FEATURES, (i + 1) * FEATURES));
        }

        X.push(reshaped);
        y.push(labelMap.get(sample.label));
    }

    const xs = tf.tensor3d(X);
    const ys = tf.oneHot(tf.tensor1d(y, 'int32'), labels.length);

    console.log(`Input shape: ${xs.shape}`);

    const model = tf.sequential();

    model.add(tf.layers.conv1d({
        inputShape: [TARGET_FRAMES, FEATURES],
        filters: 64,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same'
    }));
    model.add(tf.layers.batchNormalization());

    model.add(tf.layers.conv1d({
        filters: 128,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same'
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.maxPooling1d({ poolSize: 2 }));

    model.add(tf.layers.lstm({
        units: 128,
        returnSequences: true
    }));
    model.add(tf.layers.dropout({ rate: 0.3 }));

    model.add(tf.layers.lstm({
        units: 64,
        returnSequences: false
    }));
    model.add(tf.layers.dropout({ rate: 0.3 }));

    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: labels.length, activation: 'softmax' }));

    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    model.summary();

    console.log('Training started...');
    await model.fit(xs, ys, {
        epochs: 15,
        batchSize: 32,
        validationSplit: 0.2,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(
                    `Epoch ${epoch + 1}: Loss = ${logs?.loss?.toFixed(4)}, Acc = ${logs?.acc?.toFixed(4)}`
                );
            }
        }
    });

    if (!fs.existsSync(MODEL_SAVE_PATH)) {
        fs.mkdirSync(MODEL_SAVE_PATH, { recursive: true });
    }

    const saveResult = await model.save(tf.io.withSaveHandler(async (artifacts) => {
        const modelJSON = {
            modelTopology: artifacts.modelTopology,
            weightsManifest: [{
                paths: ['weights.bin'],
                weights: artifacts.weightSpecs
            }]
        };
        fs.writeFileSync(
            path.join(MODEL_SAVE_PATH, 'model.json'),
            JSON.stringify(modelJSON)
        );

        const weightData = Buffer.from(artifacts.weightData);
        fs.writeFileSync(
            path.join(MODEL_SAVE_PATH, 'weights.bin'),
            weightData
        );

        console.log(`Model saved to: ${MODEL_SAVE_PATH}`);
        return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
    }));

    fs.writeFileSync(
        path.join(MODEL_SAVE_PATH, 'label_map.json'),
        JSON.stringify(labels)
    );

    xs.dispose();
    ys.dispose();
}

train().catch((err) => console.error(err));
