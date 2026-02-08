import { useEffect, useRef, useState } from 'react';
import {
  HandLandmarker,
  FilesetResolver,
  type HandLandmarkerResult,
  type NormalizedLandmark
} from "@mediapipe/tasks-vision";
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';
import { HAND_CONNECTIONS } from '@mediapipe/hands';
import './index.css';

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number | null>(null);
  const lastVideoTimeRef = useRef(-1);
  const webcamRunningRef = useRef(false);
  const sampleCountsRef = useRef<Map<string, number>>(new Map());
  const missingFramesRef = useRef(0);
  const frameBufferRef = useRef<Array<Array<{ x: number, y: number, z: number }>>>([]);
  const BUFFER_SIZE = 10;

  const [handLandmarker, setHandLandmarker] = useState<HandLandmarker | null>(null);
  const [webcamRunning, setWebcamRunning] = useState(false);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [targetLabel, setTargetLabel] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [bufferCount, setBufferCount] = useState(0);

  useEffect(() => {
    const createHandLandmarker = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
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

  const enableCam = async () => {
    if (!handLandmarker) return;

    if (webcamRunning) {
      setWebcamRunning(false);
      webcamRunningRef.current = false;
      frameBufferRef.current = [];
      setBufferCount(0);
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
        const firstHand = results.landmarks[0];
        const wrist = firstHand[0];
        const normalizedFirstHand = firstHand.map(point => ({
          x: point.x - wrist.x,
          y: point.y - wrist.y,
          z: point.z - wrist.z
        }));
        frameBufferRef.current.push(normalizedFirstHand);
        if (frameBufferRef.current.length > BUFFER_SIZE) {
          frameBufferRef.current.shift();
        }
        setBufferCount(frameBufferRef.current.length);
        const averagedHand = normalizedFirstHand.map((_, pointIndex) => {
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
        const vizLandmarks = [averagedHand];
        for (const landmarks of vizLandmarks) {
          drawConnectors(canvasCtx, landmarks as NormalizedLandmark[], HAND_CONNECTIONS, {
            color: "#00FF00",
            lineWidth: 5
          });
          drawLandmarks(canvasCtx, landmarks as NormalizedLandmark[], { color: "#FF0000", lineWidth: 2 });
        }
      } else {
        missingFramesRef.current++;
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

  const startRecording = () => {
    if (!targetLabel.trim()) {
      alert("Please enter a label first.");
      return;
    }
    if (frameBufferRef.current.length < BUFFER_SIZE) {
      alert("Hand not detected or buffer not full.");
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
  };

  const saveSample = () => {
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
  };

  return (
    <div style={{ position: 'relative', padding: '20px' }}>
      <h1>Data Collection Tool</h1>
      {!isModelLoaded && <p>Loading model...</p>}
      <div style={{ marginBottom: '20px', display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
        <button
          onClick={enableCam}
          disabled={!isModelLoaded || isRecording}
          style={{ padding: '10px 20px' }}
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
            fontWeight: 'bold'
          }}
        >
          {isRecording ? "RECORDING..." : (bufferCount < BUFFER_SIZE ? `FILLING BUFFER (${bufferCount}/10)` : "REC (3s)")}
        </button>
      </div>
      <div style={{ display: 'flex', gap: '20px', flexDirection: 'row', alignItems: 'flex-start' }}>
        <div style={{ background: '#f4f4f4', padding: '15px', borderRadius: '8px', border: '1px solid #ddd', minWidth: '250px' }}>
          <h3>Status</h3>
          <p>
            <strong>Buffer State: </strong>
            <span style={{ color: bufferCount === 10 ? 'green' : 'orange', fontWeight: 'bold' }}>
              {bufferCount}/10 frames
            </span>
          </p>
          <p><strong>Last Sample:</strong><br /> {sampleCountsRef.current.get(targetLabel.trim().toLowerCase().replace(/\s+/g, '_')) ? `${targetLabel.trim().toLowerCase().replace(/\s+/g, '_')}_${sampleCountsRef.current.get(targetLabel.trim().toLowerCase().replace(/\s+/g, '_'))?.toString().padStart(3, '0')}.json` : "None"}</p>
        </div>
        <div style={{ position: 'relative', width: '640px', height: '480px', border: bufferCount === 10 ? '4px solid #4CAF50' : '4px solid #ccc' }}>
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
    </div>
  );
}

export default App;
