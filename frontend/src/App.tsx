import { useState, useCallback } from 'react';
import { Canvas } from '@react-three/fiber';
import ApartmentScene from './components/ApartmentScene';
import type { SceneObject } from './components/ApartmentScene';
import CameraRig from './components/CameraRig';
import DetectionMarker from './components/DetectionMarker';
import MemoryPanel from './components/MemoryPanel';
import ControlBar from './components/ControlBar';
import StatusToast from './components/StatusToast';
import DecisionLog from './components/DecisionLog';
import { setModelBounds, getOverviewCamera } from './config/roomPositions';
import {
  initWorld,
  resetWorld,
  stepExplore,
  autoExplore,
  searchObject,
} from './api';
import type { ApiState, SearchResponse } from './api';

interface StatusInfo {
  message: string;
  isSuccess: boolean;
  details?: string;
}

interface FlyTarget {
  position: [number, number, number];
  lookAt: [number, number, number];
}

export default function App() {
  const [state, setState] = useState<ApiState | null>(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<StatusInfo | null>(null);
  const [flyTarget, setFlyTarget] = useState<FlyTarget | null>(null);
  const [searchTarget, setSearchTarget] = useState<string | null>(null);
  const [markerPos, setMarkerPos] = useState<[number, number, number] | null>(null);
  const [markerConf, setMarkerConf] = useState(0);
  const [cameraBounds, setCameraBounds] = useState<{
    halfW: number; halfD: number; minY: number; maxY: number;
  } | null>(null);
  const [worldReady, setWorldReady] = useState(false);

  // Called when GLB finishes loading — send mesh inventory to Python backend
  const handleBoundsReady = useCallback(
    (center: [number, number, number], size: [number, number, number]) => {
      setModelBounds(center, size);
      setCameraBounds({
        halfW: size[0] / 2,
        halfD: size[2] / 2,
        minY: center[1] - size[1] * 0.3,
        maxY: center[1] + size[1],
      });
    },
    [],
  );

  const handleObjectsDiscovered = useCallback(
    async (objects: SceneObject[]) => {
      // Send the real mesh inventory to the Python backend
      const boundsMin: [number, number, number] = [
        Math.min(...objects.map((o) => o.worldPosition[0])),
        Math.min(...objects.map((o) => o.worldPosition[1])),
        Math.min(...objects.map((o) => o.worldPosition[2])),
      ];
      const boundsMax: [number, number, number] = [
        Math.max(...objects.map((o) => o.worldPosition[0])),
        Math.max(...objects.map((o) => o.worldPosition[1])),
        Math.max(...objects.map((o) => o.worldPosition[2])),
      ];

      const payload = {
        objects: objects.map((o) => ({
          name: o.name,
          cleanName: o.cleanName,
          position: o.worldPosition,
        })),
        bounds_min: boundsMin,
        bounds_max: boundsMax,
      };

      console.log(`Sending ${objects.length} objects to backend...`);
      const result = await initWorld(payload);
      console.log('Backend initialized:', result);
      setWorldReady(true);

      setStatus({
        message: `World loaded: ${result.objectCount} objects, ${result.waypointCount} waypoints`,
        isSuccess: true,
        details: 'Click Auto-Explore to start the agent.',
      });
    },
    [],
  );

  // When the 3D scene finds the searched mesh, fly camera there
  const handleObjectFound = useCallback(
    (position: [number, number, number]) => {
      setMarkerPos(position);
      setFlyTarget({
        position: [position[0] + 1.5, position[1] + 1.5, position[2] + 2],
        lookAt: position,
      });
    },
    [],
  );

  const handleReset = useCallback(async () => {
    setLoading(true);
    setStatus(null);
    setSearchTarget(null);
    setMarkerPos(null);
    const s = await resetWorld();
    setState(s);
    setFlyTarget(getOverviewCamera());
    setLoading(false);
  }, []);

  const handleStep = useCallback(async () => {
    setLoading(true);
    setSearchTarget(null);
    setMarkerPos(null);
    const s = await stepExplore();
    setState(s);
    if (s.done) {
      setStatus({ message: 'All waypoints visited!', isSuccess: true });
    }
    setLoading(false);
  }, []);

  const handleExplore = useCallback(async () => {
    setLoading(true);
    setSearchTarget(null);
    setMarkerPos(null);
    const s = await autoExplore();
    setState(s);
    setStatus({
      message: `Explored: ${s.memoryCount} objects detected across ${s.step} waypoints`,
      isSuccess: true,
      details: 'Memory built from noisy perception. Now search for an object.',
    });
    setLoading(false);
  }, []);

  const handleSearch = useCallback(
    async (query: string) => {
      setLoading(true);
      setMarkerPos(null);

      // 1. Ask backend memory for the best match
      const s: SearchResponse = await searchObject(query);
      setState(s);

      // 2. Tell the 3D scene to find + highlight the actual mesh
      setSearchTarget(query);

      const target = query.replace('find the ', '').replace('find ', '').trim();

      if (s.found && s.targetPosition) {
        setMarkerConf(s.confidence ?? 0);

        // Find how many candidates exist at different confidence levels
        const candidates = s.objects.filter((o) =>
          o.label.toLowerCase().includes(target.toLowerCase()),
        );
        const candInfo = candidates
          .map((c) => `${c.region}: ${c.confidence.toFixed(2)}`)
          .join(', ');

        setStatus({
          message: `Found "${target}" — confidence ${s.confidence?.toFixed(2)} (${candidates.length} sighting${candidates.length > 1 ? 's' : ''})`,
          isSuccess: true,
          details: candidates.length > 1
            ? `Candidates: ${candInfo}. Best match selected from accumulated evidence.`
            : `Verified through ${s.step} exploration steps with noisy perception.`,
        });
      } else {
        setStatus({
          message: s.message || `"${target}" not found with sufficient confidence`,
          isSuccess: false,
          details: 'Try exploring more, or search for a different object.',
        });
      }
      setLoading(false);
    },
    [],
  );

  return (
    <div className="w-screen h-screen bg-[#080b10] relative">
      {/* Title */}
      <div className="absolute top-4 left-1/2 -translate-x-1/2 z-10 px-5 py-2 rounded-xl border border-white/10 bg-[#0d1117]/80 backdrop-blur-xl text-center">
        <h1 className="text-sm font-bold tracking-widest text-amber-400">SCOUTMEM-X</h1>
        <p className="text-[10px] text-gray-500 mt-0.5">Active Scene Memory — noisy perception, real confidence</p>
      </div>

      {/* 3D Canvas */}
      <Canvas
        shadows
        camera={{ fov: 50, near: 0.1, far: 200 }}
        gl={{ antialias: true, toneMapping: 3, toneMappingExposure: 1.2 }}
      >
        <fog attach="fog" args={['#080b10', 30, 80]} />
        <ApartmentScene
          onBoundsReady={handleBoundsReady}
          onObjectsDiscovered={handleObjectsDiscovered}
          searchTarget={searchTarget}
          onObjectFound={handleObjectFound}
        />
        {markerPos && (
          <>
            <DetectionMarker position={markerPos} label="" confidence={markerConf} />
            <pointLight
              position={[markerPos[0], markerPos[1] + 1, markerPos[2]]}
              color="#3fb950"
              intensity={8}
              distance={4}
            />
          </>
        )}
        <CameraRig flyTarget={flyTarget} bounds={cameraBounds ?? undefined} />
      </Canvas>

      {/* UI Overlays */}
      <MemoryPanel state={state} />
      <DecisionLog log={state?.log ?? []} />
      <ControlBar
        onReset={handleReset}
        onStep={handleStep}
        onExplore={handleExplore}
        onSearch={handleSearch}
        loading={loading}
        allExplored={state?.allExplored ?? false}
      />
      {status && (
        <StatusToast
          message={status.message}
          isSuccess={status.isSuccess}
          details={status.details}
          onDismiss={() => setStatus(null)}
        />
      )}
    </div>
  );
}
