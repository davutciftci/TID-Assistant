import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as path from 'path';
import { getModel, getLabelMap } from './predict';

const FEEDBACK_DIR = path.join(__dirname, '..', 'feedback_dataset');
const FINE_TUNE_THRESHOLD = 10;
const TARGET_FRAMES = 30;
const FEATURES = 126;

interface FeedbackEntry {
    keypoints: number[];
    frames: number;
    correctLabel: string;
    wasCorrect: boolean;
    predictedLabel: string;
    confidence: number;
    timestamp: number;
}

let feedbackBuffer: FeedbackEntry[] = [];
let totalFeedbackCount = 0;

export function initFeedbackDir(): void {
    if (!fs.existsSync(FEEDBACK_DIR)) {
        fs.mkdirSync(FEEDBACK_DIR, { recursive: true });
    }
}

export function addFeedback(entry: FeedbackEntry): { saved: boolean; bufferSize: number; willFineTune: boolean } {
    feedbackBuffer.push(entry);
    totalFeedbackCount++;

    const filename = `feedback_${totalFeedbackCount.toString().padStart(5, '0')}.json`;
    fs.writeFileSync(
        path.join(FEEDBACK_DIR, filename),
        JSON.stringify(entry, null, 2)
    );

    const willFineTune = feedbackBuffer.length >= FINE_TUNE_THRESHOLD;

    return {
        saved: true,
        bufferSize: feedbackBuffer.length,
        willFineTune
    };
}

function normalizeForTraining(keypoints: number[], frames: number): number[] {
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

export async function fineTuneModel(): Promise<{ success: boolean; message: string }> {
    const model = getModel();
    const labelMap = getLabelMap();

    if (!model || labelMap.length === 0) {
        return { success: false, message: 'Model not loaded' };
    }

    if (feedbackBuffer.length === 0) {
        return { success: false, message: 'No feedback data available' };
    }

    console.log(`Fine-tuning with ${feedbackBuffer.length} feedback samples...`);

    const X: number[][][] = [];
    const y: number[] = [];

    for (const entry of feedbackBuffer) {
        const labelIndex = labelMap.indexOf(entry.correctLabel);
        if (labelIndex === -1) continue;

        const normalized = normalizeForTraining(entry.keypoints, entry.frames);

        let processedKeypoints: number[];
        if (entry.frames >= TARGET_FRAMES) {
            processedKeypoints = normalized.slice(0, TARGET_FRAMES * FEATURES);
        } else {
            processedKeypoints = [...normalized];
            const padding = new Array((TARGET_FRAMES - entry.frames) * FEATURES).fill(0);
            processedKeypoints.push(...padding);
        }

        const reshaped: number[][] = [];
        for (let i = 0; i < TARGET_FRAMES; i++) {
            reshaped.push(processedKeypoints.slice(i * FEATURES, (i + 1) * FEATURES));
        }

        const repeatCount = entry.wasCorrect ? 1 : 3;
        for (let r = 0; r < repeatCount; r++) {
            X.push(reshaped);
            y.push(labelIndex);
        }
    }

    if (X.length === 0) {
        return { success: false, message: 'No valid feedback samples to train on' };
    }

    const xs = tf.tensor3d(X);
    const ys = tf.oneHot(tf.tensor1d(y, 'int32'), labelMap.length);

    model.compile({
        optimizer: tf.train.adam(0.0001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    await model.fit(xs, ys, {
        epochs: 5,
        batchSize: 16,
        callbacks: {
            onEpochEnd: (epoch: number, logs?: tf.Logs) => {
                console.log(
                    `Fine-tune Epoch ${epoch + 1}: Loss = ${logs?.loss?.toFixed(4)}, Acc = ${logs?.acc?.toFixed(4)}`
                );
            }
        }
    });

    const modelSavePath = path.join(__dirname, '..', 'model');
    await model.save(tf.io.withSaveHandler(async (artifacts) => {
        const modelJSON = {
            modelTopology: artifacts.modelTopology,
            weightsManifest: [{
                paths: ['weights.bin'],
                weights: artifacts.weightSpecs
            }]
        };
        fs.writeFileSync(
            path.join(modelSavePath, 'model.json'),
            JSON.stringify(modelJSON)
        );
        const weightData = Buffer.from(artifacts.weightData as ArrayBuffer);
        fs.writeFileSync(
            path.join(modelSavePath, 'weights.bin'),
            weightData
        );
        return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
    }));
    console.log('Fine-tuned model saved.');

    xs.dispose();
    ys.dispose();

    feedbackBuffer = [];

    return {
        success: true,
        message: `Fine-tuning complete with ${X.length} weighted samples.`
    };
}

export function getFeedbackStats(): {
    bufferSize: number;
    totalFeedback: number;
    threshold: number;
} {
    return {
        bufferSize: feedbackBuffer.length,
        totalFeedback: totalFeedbackCount,
        threshold: FINE_TUNE_THRESHOLD
    };
}
