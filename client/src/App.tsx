import { useEffect, useRef, useState, useCallback } from 'react';
import {
  HandLandmarker,
  FilesetResolver,
  type HandLandmarkerResult,
  type NormalizedLandmark
} from "@mediapipe/tasks-vision";
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';
import { HAND_CONNECTIONS } from '@mediapipe/hands';

type Point3D = { x: number; y: number; z: number };
const ZERO_HAND: Point3D[] = Array.from({ length: 21 }, () => ({ x: 0, y: 0, z: 0 }));

const API_URL = 'http://localhost:3000';

interface AIPrediction {
  label: string;
  confidence: number;
  allPredictions: Array<{ label: string; confidence: number }>;
}

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number | null>(null);
  const lastVideoTimeRef = useRef(-1);
  const webcamRunningRef = useRef(false);
  const sampleCountsRef = useRef<Map<string, number>>(new Map());
  const missingFramesRef = useRef(0);
  const frameBufferRef = useRef<Array<Point3D[]>>([]);
  const lastWristPosRef = useRef<Point3D | null>(null);
  const isStableRef = useRef(false);
  const unstableCountRef = useRef(0);
  const lastPredictionTimeRef = useRef(0);
  const lastKeypointsRef = useRef<number[] | null>(null);
  const isAIModeRef = useRef(false);
  const BUFFER_SIZE = 20;
  const POINTS_PER_FRAME = 42;
  const STABILITY_THRESHOLD = 0.09;
  const UNSTABLE_RESET_THRESHOLD = 15;
  const PREDICTION_INTERVAL = 500;

  const [handLandmarker, setHandLandmarker] = useState<HandLandmarker | null>(null);
  const [webcamRunning, setWebcamRunning] = useState(false);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [targetLabel, setTargetLabel] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [bufferCount, setBufferCount] = useState(0);
  const [isAutoMode, setIsAutoMode] = useState(false);
  const [isAIMode, setIsAIMode] = useState(false);

  useEffect(() => {
    isAIModeRef.current = isAIMode;
  }, [isAIMode]);
  const [aiPrediction, setAIPrediction] = useState<AIPrediction | null>(null);
  const [isStable, setIsStable] = useState(false);
  const [detectedHands, setDetectedHands] = useState(0);
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [showCorrectionModal, setShowCorrectionModal] = useState(false);
  const [correctionLabel, setCorrectionLabel] = useState("");
  const [feedbackStats, setFeedbackStats] = useState<{ bufferSize: number; totalFeedback: number } | null>(null);

  useEffect(() => {
    const createHandLandmarker = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
        );
        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 2
        });
        setHandLandmarker(landmarker);
        setIsModelLoaded(true);
      } catch (error) {
        console.error("Error creating hand landmarker:", error);
      }
    };
    createHandLandmarker();
    return () => {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
    }
  }, []);

  const sendPrediction = useCallback(async () => {
    if (!frameBufferRef.current.length || frameBufferRef.current.length < BUFFER_SIZE) return;
    if (!isStableRef.current) return;

    const now = Date.now();
    if (now - lastPredictionTimeRef.current < PREDICTION_INTERVAL) return;
    lastPredictionTimeRef.current = now;

    const flattenedVector = frameBufferRef.current.flatMap(frame =>
      frame.flatMap(p => [p.x, p.y, p.z])
    );

    lastKeypointsRef.current = flattenedVector;

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          keypoints: flattenedVector,
          frames: BUFFER_SIZE
        })
      });

      if (response.ok) {
        const result: AIPrediction = await response.json();
        setAIPrediction(result);
        setFeedbackSent(false);
      }
    } catch (error) {
      console.error('Prediction API error:', error);
    }
  }, []);

  const sendFeedback = useCallback(async (wasCorrect: boolean, correctLabel?: string) => {
    if (!aiPrediction || !lastKeypointsRef.current) return;

    try {
      const response = await fetch(`${API_URL}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          keypoints: lastKeypointsRef.current,
          frames: BUFFER_SIZE,
          correctLabel: wasCorrect ? aiPrediction.label : (correctLabel || 'unknown'),
          wasCorrect,
          predictedLabel: aiPrediction.label,
          confidence: aiPrediction.confidence
        })
      });

      if (response.ok) {
        const result = await response.json();
        setFeedbackSent(true);
        setFeedbackStats({ bufferSize: result.bufferSize, totalFeedback: result.totalFeedback || 0 });
        if (result.fineTuneResult) {
          console.log('Model fine-tuned:', result.fineTuneResult.message);
        }
      }
    } catch (error) {
      console.error('Feedback API error:', error);
    }
  }, [aiPrediction]);

  const enableCam = async () => {
    if (!handLandmarker) return;

    if (webcamRunning) {
      setWebcamRunning(false);
      webcamRunningRef.current = false;
      frameBufferRef.current = [];
      setBufferCount(0);
      setDetectedHands(0);
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
        videoRef.current.srcObject = null;
      }
    } else {
      setWebcamRunning(true);
      webcamRunningRef.current = true;
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.addEventListener("loadeddata", predictWebcam);
        }
      } catch (err) {
        console.error("Error accessing webcam:", err);
        setWebcamRunning(false);
        webcamRunningRef.current = false;
      }
    }
  };

  const predictWebcam = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !handLandmarker || !webcamRunningRef.current) return;

    if (video.videoWidth > 0 && video.videoHeight > 0) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const startTimeMs = performance.now();
    let results: HandLandmarkerResult | null = null;
    if (lastVideoTimeRef.current !== video.currentTime) {
      lastVideoTimeRef.current = video.currentTime;
      results = handLandmarker.detectForVideo(video, startTimeMs);
    }

    const canvasCtx = canvas.getContext("2d");
    if (canvasCtx) {
      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
      if (results && results.landmarks && results.landmarks.length > 0) {
        missingFramesRef.current = 0;
        const numHands = results.landmarks.length;
        setDetectedHands(numHands);

        const firstHand = results.landmarks[0];
        const wrist = firstHand[0];

        if (lastWristPosRef.current) {
          const dx = wrist.x - lastWristPosRef.current.x;
          const dy = wrist.y - lastWristPosRef.current.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          const stable = dist < STABILITY_THRESHOLD;
          isStableRef.current = stable;
          setIsStable(stable);
          if (!stable) {
            unstableCountRef.current++;
          } else {
            unstableCountRef.current = 0;
          }
        }
        lastWristPosRef.current = { x: wrist.x, y: wrist.y, z: wrist.z };

        const normalizedHand1 = firstHand.map(point => ({
          x: point.x - wrist.x,
          y: point.y - wrist.y,
          z: point.z - wrist.z
        }));

        let normalizedHand2: Point3D[];
        if (numHands >= 2) {
          const secondHand = results.landmarks[1];
          normalizedHand2 = secondHand.map(point => ({
            x: point.x - wrist.x,
            y: point.y - wrist.y,
            z: point.z - wrist.z
          }));
        } else {
          normalizedHand2 = ZERO_HAND;
        }

        const combinedFrame: Point3D[] = [...normalizedHand1, ...normalizedHand2];

        frameBufferRef.current.push(combinedFrame);
        if (frameBufferRef.current.length > BUFFER_SIZE) {
          frameBufferRef.current.shift();
        }
        setBufferCount(frameBufferRef.current.length);

        const averagedFrame = combinedFrame.map((_, pointIndex) => {
          let sumX = 0, sumY = 0, sumZ = 0;
          frameBufferRef.current.forEach(frame => {
            sumX += frame[pointIndex].x;
            sumY += frame[pointIndex].y;
            sumZ += frame[pointIndex].z;
          });
          const count = frameBufferRef.current.length;
          return {
            x: sumX / count,
            y: sumY / count,
            z: sumZ / count
          };
        });

        if (isAIModeRef.current && frameBufferRef.current.length === BUFFER_SIZE && isStableRef.current) {
          sendPrediction();
        } else if (!isAIModeRef.current) {
          setAIPrediction(null);
        }

        if (unstableCountRef.current > UNSTABLE_RESET_THRESHOLD && isAIModeRef.current) {
          setAIPrediction(null);
        }

        const vizHand1 = averagedFrame.slice(0, 21);
        drawConnectors(canvasCtx, vizHand1 as NormalizedLandmark[], HAND_CONNECTIONS, {
          color: "#00FF00",
          lineWidth: 5
        });
        drawLandmarks(canvasCtx, vizHand1 as NormalizedLandmark[], { color: "#FF0000", lineWidth: 2 });

        if (numHands >= 2) {
          const vizHand2 = averagedFrame.slice(21, 42);
          drawConnectors(canvasCtx, vizHand2 as NormalizedLandmark[], HAND_CONNECTIONS, {
            color: "#00AAFF",
            lineWidth: 5
          });
          drawLandmarks(canvasCtx, vizHand2 as NormalizedLandmark[], { color: "#FF6600", lineWidth: 2 });
        }
      } else {
        missingFramesRef.current++;
        setDetectedHands(0);
        if (missingFramesRef.current > 5) {
          frameBufferRef.current = [];
          setBufferCount(0);
        }
      }
      canvasCtx.restore();
    }
    if (webcamRunningRef.current) {
      requestRef.current = requestAnimationFrame(predictWebcam);
    }
  };

  const saveSample = useCallback(() => {
    if (frameBufferRef.current.length < BUFFER_SIZE) return;
    const label = targetLabel.trim().toLowerCase().replace(/\s+/g, '_');
    const currentCount = sampleCountsRef.current.get(label) || 0;
    const nextId = currentCount + 1;
    sampleCountsRef.current.set(label, nextId);
    const filename = `${label}_${nextId.toString().padStart(3, '0')}.json`;
    const flattenedVector = frameBufferRef.current.flatMap(frame =>
      frame.flatMap(p => [p.x, p.y, p.z])
    );
    const sampleData = {
      label: label,
      frames: BUFFER_SIZE,
      pointsPerFrame: POINTS_PER_FRAME,
      keypoints: flattenedVector
    };
    const blob = new Blob([JSON.stringify(sampleData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    alert(`Kayıt Başarılı: ${filename}`);
    frameBufferRef.current = [];
    setBufferCount(0);
  }, [targetLabel]);

  const startRecording = useCallback(() => {
    if (!targetLabel.trim()) {
      return;
    }
    if (frameBufferRef.current.length < BUFFER_SIZE) {
      if (!isAutoMode) alert("Hand not detected or buffer not full.");
      return;
    }
    setIsRecording(true);
    let count = 3;
    setCountdown(count);
    const interval = setInterval(() => {
      count--;
      if (count > 0) {
        setCountdown(count);
      } else {
        clearInterval(interval);
        setCountdown(null);
        saveSample();
        setIsRecording(false);
      }
    }, 1000);
  }, [targetLabel, isAutoMode, saveSample]);

  useEffect(() => {
    if (isAutoMode && bufferCount === BUFFER_SIZE && !isRecording && countdown === null && targetLabel.trim()) {
      startRecording();
    }
  }, [isAutoMode, bufferCount, isRecording, countdown, targetLabel, startRecording]);

  return (
    <div style={{ position: 'relative', padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1>TİD Asistanı</h1>
      {!isModelLoaded && <p>Loading model...</p>}
      <div style={{ marginBottom: '20px', display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
        <button
          onClick={enableCam}
          disabled={!isModelLoaded || isRecording}
          style={{ padding: '10px 20px', cursor: 'pointer' }}
        >
          {webcamRunning ? "STOP CAMERA" : "START CAMERA"}
        </button>
        <input
          type="text"
          placeholder="Enter Label"
          value={targetLabel}
          onChange={(e) => setTargetLabel(e.target.value)}
          disabled={isRecording}
          style={{ padding: '10px', fontSize: '16px' }}
        />
        <button
          onClick={startRecording}
          disabled={!isModelLoaded || !webcamRunning || isRecording || bufferCount < BUFFER_SIZE}
          style={{
            padding: '10px 20px',
            backgroundColor: (isRecording || bufferCount < BUFFER_SIZE) ? '#ccc' : '#f44336',
            color: 'white',
            border: 'none',
            fontWeight: 'bold',
            cursor: 'pointer'
          }}
        >
          {isRecording ? "RECORDING..." : (bufferCount < BUFFER_SIZE ? `FILLING BUFFER (${bufferCount}/20)` : "REC (3s)")}
        </button>
        <label style={{ display: 'flex', alignItems: 'center', gap: '5px', cursor: 'pointer', background: isAutoMode ? '#4CAF50' : '#e0e0e0', color: isAutoMode ? 'white' : 'black', padding: '10px', borderRadius: '5px', transition: '0.3s' }}>
          <input
            type="checkbox"
            checked={isAutoMode}
            onChange={(e) => setIsAutoMode(e.target.checked)}
          />
          <span style={{ fontWeight: 'bold' }}>AUTO REC</span>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '5px', cursor: 'pointer', background: isAIMode ? '#9C27B0' : '#e0e0e0', color: isAIMode ? 'white' : 'black', padding: '10px', borderRadius: '5px', transition: '0.3s' }}>
          <input
            type="checkbox"
            checked={isAIMode}
            onChange={(e) => {
              setIsAIMode(e.target.checked);
              if (!e.target.checked) {
                setAIPrediction(null);
                setFeedbackSent(false);
              }
            }}
          />
          <span style={{ fontWeight: 'bold' }}>🤖 AI MODE</span>
        </label>
      </div>
      <div style={{ display: 'flex', gap: '20px', flexDirection: 'row', alignItems: 'flex-start' }}>
        <div style={{ background: '#f4f4f4', padding: '15px', borderRadius: '8px', border: '1px solid #ddd', minWidth: '280px' }}>
          <h3 style={{ marginTop: 0 }}>Status</h3>
          <p>
            <strong>Hands Detected:</strong>{' '}
            <span style={{ color: detectedHands > 0 ? 'green' : 'red', fontWeight: 'bold' }}>
              {detectedHands === 0 ? 'None' : `${detectedHands} hand${detectedHands > 1 ? 's' : ''}`}
            </span>
          </p>
          <p>
            <strong>Buffer State:</strong>{' '}
            <span style={{ color: bufferCount === 20 ? 'green' : 'orange', fontWeight: 'bold' }}>
              {bufferCount}/20 frames
            </span>
          </p>
          <p>
            <strong>Stability:</strong>{' '}
            <span style={{ color: isStable ? 'green' : 'red', fontWeight: 'bold' }}>
              {isStable ? 'STABLE' : 'MOVING...'}
            </span>
          </p>

          {isAIMode && aiPrediction && (
            <div style={{ marginTop: '20px', padding: '15px', background: '#fff', borderRadius: '8px', border: '2px solid #9C27B0', boxShadow: '0 2px 8px rgba(156,39,176,0.2)' }}>
              <div style={{ color: '#9C27B0', fontSize: '0.8em', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: 'bold', marginBottom: '5px' }}>
                🤖 AI Tahmini
              </div>
              <div style={{ fontSize: '2.5em', color: '#333', fontWeight: 'bold', textAlign: 'center' }}>
                {aiPrediction.label.toUpperCase()}
              </div>
              <div style={{ marginTop: '10px', height: '10px', background: '#eee', borderRadius: '5px', overflow: 'hidden' }}>
                <div style={{
                  height: '100%',
                  background: aiPrediction.confidence > 0.8 ? '#4CAF50' : aiPrediction.confidence > 0.5 ? '#FF9800' : '#f44336',
                  width: `${(aiPrediction.confidence * 100)}%`,
                  transition: '0.3s'
                }}></div>
              </div>
              <small style={{ color: '#666' }}>
                Güven: {(aiPrediction.confidence * 100).toFixed(1)}%
              </small>

              {aiPrediction.allPredictions && aiPrediction.allPredictions.length > 1 && (
                <div style={{ marginTop: '10px', fontSize: '0.85em', color: '#888' }}>
                  {aiPrediction.allPredictions.slice(1).map((pred, i) => (
                    <div key={i}>{pred.label.toUpperCase()}: {(pred.confidence * 100).toFixed(1)}%</div>
                  ))}
                </div>
              )}

              {!feedbackSent ? (
                <div style={{ marginTop: '15px', display: 'flex', gap: '10px' }}>
                  <button
                    onClick={() => sendFeedback(true)}
                    style={{
                      flex: 1, padding: '10px', background: '#4CAF50', color: 'white',
                      border: 'none', borderRadius: '5px', fontWeight: 'bold',
                      cursor: 'pointer', fontSize: '1.1em'
                    }}
                  >
                    ✓ Doğru
                  </button>
                  <button
                    onClick={() => setShowCorrectionModal(true)}
                    style={{
                      flex: 1, padding: '10px', background: '#f44336', color: 'white',
                      border: 'none', borderRadius: '5px', fontWeight: 'bold',
                      cursor: 'pointer', fontSize: '1.1em'
                    }}
                  >
                    ✗ Yanlış
                  </button>
                </div>
              ) : (
                <div style={{ marginTop: '15px', textAlign: 'center', color: '#4CAF50', fontWeight: 'bold' }}>
                  ✓ Geri bildirim kaydedildi
                </div>
              )}

              {feedbackStats && (
                <div style={{ marginTop: '8px', fontSize: '0.75em', color: '#999', textAlign: 'center' }}>
                  Toplam: {feedbackStats.totalFeedback} feedback | Buffer: {feedbackStats.bufferSize}/10
                </div>
              )}
            </div>
          )}
        </div>
        <div style={{ position: 'relative', width: '640px', height: '480px', border: bufferCount === 20 ? '4px solid #4CAF50' : '4px solid #ccc', borderRadius: '8px', overflow: 'hidden' }}>
          <video ref={videoRef} autoPlay playsInline style={{ width: '100%', height: '100%', objectFit: 'cover' }}></video>
          <canvas ref={canvasRef} style={{ position: 'absolute', left: 0, top: 0, width: '100%', height: '100%' }}></canvas>
          {countdown !== null && (
            <div style={{
              position: 'absolute', top: 0, left: 0, width: '100%', height: '100%',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              backgroundColor: 'rgba(0,0,0,0.5)', color: 'white', fontSize: '100px', fontWeight: 'bold', zIndex: 10
            }}>
              {countdown}
            </div>
          )}
        </div>
      </div>

      {showCorrectionModal && (
        <div style={{
          position: 'fixed', top: 0, left: 0, width: '100%', height: '100%',
          backgroundColor: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center',
          justifyContent: 'center', zIndex: 100
        }}>
          <div style={{
            background: 'white', padding: '30px', borderRadius: '12px',
            boxShadow: '0 4px 20px rgba(0,0,0,0.3)', minWidth: '300px'
          }}>
            <h3 style={{ marginTop: 0 }}>Doğru İşaret Neydi?</h3>
            <p style={{ color: '#666' }}>
              AI <strong>{aiPrediction?.label.toUpperCase()}</strong> tahmin etti. Doğru etiketi girin:
            </p>
            <input
              type="text"
              placeholder="Doğru etiket (ör: a, b, merhaba)"
              value={correctionLabel}
              onChange={(e) => setCorrectionLabel(e.target.value)}
              style={{ width: '100%', padding: '10px', fontSize: '16px', borderRadius: '5px', border: '1px solid #ddd', boxSizing: 'border-box' }}
              autoFocus
            />
            <div style={{ marginTop: '15px', display: 'flex', gap: '10px' }}>
              <button
                onClick={() => {
                  if (correctionLabel.trim()) {
                    sendFeedback(false, correctionLabel.trim().toLowerCase());
                    setShowCorrectionModal(false);
                    setCorrectionLabel("");
                  }
                }}
                disabled={!correctionLabel.trim()}
                style={{
                  flex: 1, padding: '10px', background: '#9C27B0', color: 'white',
                  border: 'none', borderRadius: '5px', fontWeight: 'bold', cursor: 'pointer'
                }}
              >
                Gönder
              </button>
              <button
                onClick={() => { setShowCorrectionModal(false); setCorrectionLabel(""); }}
                style={{
                  flex: 1, padding: '10px', background: '#e0e0e0', color: '#333',
                  border: 'none', borderRadius: '5px', fontWeight: 'bold', cursor: 'pointer'
                }}
              >
                İptal
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
