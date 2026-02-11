import * as fs from 'fs';
import * as path from 'path';

interface LandmarkData {
    label: string;
    frames: number;
    pointsPerFrame: number;
    keypoints: number[];
}

const DATASET_PATH = path.join(process.cwd(), 'dataset');
const OUTPUT_PATH = path.join(process.cwd(), 'dataset_processed.json');

function normalizeLandmarks(keypoints: number[], frames: number, pointsPerFrame: number) {
    const normalized = [];
    const valuesPerFrame = pointsPerFrame * 3;

    for (let f = 0; f < frames; f++) {
        const frameStart = f * valuesPerFrame;
        
        for (let hand = 0; hand < 2; hand++) {
            const handStart = frameStart + hand * 21 * 3;
            
            const wristX = keypoints[handStart];
            const wristY = keypoints[handStart + 1];
            const wristZ = keypoints[handStart + 2];

            const mcpX = keypoints[handStart + 9 * 3];
            const mcpY = keypoints[handStart + 9 * 3 + 1];
            const mcpZ = keypoints[handStart + 9 * 3 + 2];
            
            const distance = Math.sqrt(
                Math.pow(mcpX - wristX, 2) + 
                Math.pow(mcpY - wristY, 2) + 
                Math.pow(mcpZ - wristZ, 2)
            ) || 1.0;

            for (let i = 0; i < 21; i++) {
                const idx = handStart + i * 3;
                
                normalized.push((keypoints[idx] - wristX) / distance);
                normalized.push((keypoints[idx + 1] - wristY) / distance);
                normalized.push((keypoints[idx + 2] - wristZ) / distance);
            }
        }
    }
    return normalized;
}

function processDirectory(dir: string, allData: any[]) {
    const items = fs.readdirSync(dir);

    for (const item of items) {
        const fullPath = path.join(dir, item);
        const stat = fs.statSync(fullPath);

        if (stat.isDirectory()) {
            processDirectory(fullPath, allData);
        } else if (item.endsWith('.json')) {
            const content = fs.readFileSync(fullPath, 'utf8');
            const data: LandmarkData = JSON.parse(content);
            
            const normalizedPoints = normalizeLandmarks(data.keypoints, data.frames, data.pointsPerFrame);
            
            allData.push({
                label: data.label,
                frames: data.frames,
                keypoints: normalizedPoints
            });
            console.log(`Processed: ${item}`);
        }
    }
}

const finalDataset: any[] = [];
console.log('Starting preprocessing...');
processDirectory(DATASET_PATH, finalDataset);

fs.writeFileSync(OUTPUT_PATH, JSON.stringify(finalDataset));
console.log(`Preprocessing complete. Total samples: ${finalDataset.length}`);
console.log(`Saved to: ${OUTPUT_PATH}`);
