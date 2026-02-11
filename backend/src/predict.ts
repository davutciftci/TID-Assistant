import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as path from 'path';

const MODEL_PATH = path.join(__dirname, '..', 'model');
const TARGET_FRAMES = 30;
const FEATURES = 126;

let model: tf.LayersModel | null = null;
let labelMap: string[] = [];

export async function loadModel(): Promise<void> {
    const modelJsonPath = path.join(MODEL_PATH, 'model.json');
    const labelMapPath = path.join(MODEL_PATH, 'label_map.json');

    if (!fs.existsSync(modelJsonPath)) {
        console.warn('Model not found. Please train the model first.');
        return;
    }

    const modelJSON = JSON.parse(fs.readFileSync(modelJsonPath, 'utf8'));
    const weightsManifest = modelJSON.weightsManifest;

    model = await tf.loadLayersModel(tf.io.fromMemory(
        modelJSON.modelTopology,
        weightsManifest[0].weights,
        fs.readFileSync(path.join(MODEL_PATH, weightsManifest[0].paths[0])).buffer
    ));
    console.log('TID Model loaded successfully.');

    if (fs.existsSync(labelMapPath)) {
        labelMap = JSON.parse(fs.readFileSync(labelMapPath, 'utf8'));
        console.log(`Label map loaded: ${labelMap.join(', ')}`);
    }
}

function normalizeKeypoints(keypoints: number[], frames: number): number[] {
    const normalized: number[] = [];
    const valuesPerFrame = 42 * 3;

    for (let f = 0; f < frames; f++) {
        const frameStart = f * valuesPerFrame;

        for (let hand = 0; hand < 2; hand++) {
            const handStart = frameStart + hand * 21 * 3;

            const wristX = keypoints[handStart] || 0;
            const wristY = keypoints[handStart + 1] || 0;
            const wristZ = keypoints[handStart + 2] || 0;

            const mcpX = keypoints[handStart + 9 * 3] || 0;
            const mcpY = keypoints[handStart + 9 * 3 + 1] || 0;
            const mcpZ = keypoints[handStart + 9 * 3 + 2] || 0;

            const distance = Math.sqrt(
                Math.pow(mcpX - wristX, 2) +
                Math.pow(mcpY - wristY, 2) +
                Math.pow(mcpZ - wristZ, 2)
            ) || 1.0;

            for (let i = 0; i < 21; i++) {
                const idx = handStart + i * 3;
                normalized.push(((keypoints[idx] || 0) - wristX) / distance);
                normalized.push(((keypoints[idx + 1] || 0) - wristY) / distance);
                normalized.push(((keypoints[idx + 2] || 0) - wristZ) / distance);
            }
        }
    }
    return normalized;
}

export interface PredictionResult {
    label: string;
    confidence: number;
    allPredictions: Array<{ label: string; confidence: number }>;
}

export async function predict(keypoints: number[], frames: number): Promise<PredictionResult | null> {
    if (!model || labelMap.length === 0) {
        return null;
    }

    const normalized = normalizeKeypoints(keypoints, frames);

    let processedKeypoints: number[];
    if (frames >= TARGET_FRAMES) {
        processedKeypoints = normalized.slice(0, TARGET_FRAMES * FEATURES);
    } else {
        processedKeypoints = [...normalized];
        const padding = new Array((TARGET_FRAMES - frames) * FEATURES).fill(0);
        processedKeypoints.push(...padding);
    }

    const reshaped: number[][] = [];
    for (let i = 0; i < TARGET_FRAMES; i++) {
        reshaped.push(processedKeypoints.slice(i * FEATURES, (i + 1) * FEATURES));
    }

    const inputTensor = tf.tensor3d([reshaped]);
    const prediction = model.predict(inputTensor) as tf.Tensor;
    const probabilities = await prediction.data();

    inputTensor.dispose();
    prediction.dispose();

    const allPredictions = labelMap.map((label, index) => ({
        label,
        confidence: probabilities[index]
    })).sort((a, b) => b.confidence - a.confidence);

    const topPrediction = allPredictions[0];

    return {
        label: topPrediction.label,
        confidence: topPrediction.confidence,
        allPredictions: allPredictions.slice(0, 3)
    };
}

export function getModelInfo(): { loaded: boolean; classes: string[]; classCount: number } {
    return {
        loaded: model !== null,
        classes: labelMap,
        classCount: labelMap.length
    };
}

export function getModel(): tf.LayersModel | null {
    return model;
}

export function getLabelMap(): string[] {
    return labelMap;
}
