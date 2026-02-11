import 'dotenv/config';
import express, { Request, Response } from 'express';
import cors from 'cors';
import { loadModel, predict, getModelInfo } from './src/predict';
import { initFeedbackDir, addFeedback, fineTuneModel, getFeedbackStats } from './src/reinforcement';

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '10mb' }));

app.get('/', (req: Request, res: Response) => {
  res.send('TID Assistant API');
});

app.post('/predict', async (req: Request, res: Response) => {
  try {
    const { keypoints, frames } = req.body;

    if (!keypoints || !frames) {
      res.status(400).json({ error: 'keypoints and frames are required' });
      return;
    }

    const result = await predict(keypoints, frames);

    if (!result) {
      res.status(503).json({ error: 'Model not loaded. Train the model first.' });
      return;
    }

    res.json(result);
  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({ error: 'Prediction failed' });
  }
});

app.post('/feedback', async (req: Request, res: Response) => {
  try {
    const { keypoints, frames, correctLabel, wasCorrect, predictedLabel, confidence } = req.body;

    if (!keypoints || !frames || !correctLabel || wasCorrect === undefined) {
      res.status(400).json({ error: 'keypoints, frames, correctLabel, and wasCorrect are required' });
      return;
    }

    const result = addFeedback({
      keypoints,
      frames,
      correctLabel,
      wasCorrect,
      predictedLabel: predictedLabel || 'unknown',
      confidence: confidence || 0,
      timestamp: Date.now()
    });

    if (result.willFineTune) {
      const fineTuneResult = await fineTuneModel();
      res.json({ ...result, fineTuneResult });
      return;
    }

    res.json(result);
  } catch (error) {
    console.error('Feedback error:', error);
    res.status(500).json({ error: 'Feedback failed' });
  }
});

app.get('/model/info', (req: Request, res: Response) => {
  const modelInfo = getModelInfo();
  const feedbackStats = getFeedbackStats();
  res.json({ ...modelInfo, feedback: feedbackStats });
});

app.post('/model/fine-tune', async (req: Request, res: Response) => {
  try {
    const result = await fineTuneModel();
    res.json(result);
  } catch (error) {
    console.error('Fine-tune error:', error);
    res.status(500).json({ error: 'Fine-tuning failed' });
  }
});

async function startServer() {
  initFeedbackDir();
  await loadModel();

  const server = app.listen(port, () => {
    console.log(`TID Assistant API running on port ${port}`);
  });

  server.on('error', (err: NodeJS.ErrnoException) => {
    if (err.code === 'EADDRINUSE') {
      console.error(`Port ${port} is already in use.`);
      process.exit(1);
    } else {
      console.error('Server error:', err);
    }
  });
}

startServer();
