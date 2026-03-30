import { useRef, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { Vector3 } from 'three';

interface FlyTarget {
  position: [number, number, number];
  lookAt: [number, number, number];
}

interface Props {
  flyTarget: FlyTarget | null;
  bounds?: { halfW: number; halfD: number; minY: number; maxY: number };
  onFlyComplete?: () => void;
}

export default function CameraRig({ flyTarget, bounds, onFlyComplete }: Props) {
  const controlsRef = useRef<any>(null);
  const { camera } = useThree();
  const flyRef = useRef<{
    startPos: Vector3;
    endPos: Vector3;
    startTarget: Vector3;
    endTarget: Vector3;
    progress: number;
  } | null>(null);

  useEffect(() => {
    if (flyTarget && controlsRef.current) {
      flyRef.current = {
        startPos: camera.position.clone(),
        endPos: new Vector3(...flyTarget.position),
        startTarget: controlsRef.current.target.clone(),
        endTarget: new Vector3(...flyTarget.lookAt),
        progress: 0,
      };
    }
  }, [flyTarget, camera]);

  useFrame(() => {
    // Fly animation
    if (flyRef.current && controlsRef.current) {
      const fly = flyRef.current;
      fly.progress = Math.min(fly.progress + 0.015, 1);
      const t = fly.progress * fly.progress * (3 - 2 * fly.progress);
      camera.position.lerpVectors(fly.startPos, fly.endPos, t);
      controlsRef.current.target.lerpVectors(fly.startTarget, fly.endTarget, t);
      controlsRef.current.update();
      if (fly.progress >= 1) {
        flyRef.current = null;
        onFlyComplete?.();
      }
    }

    // Clamp camera and orbit target inside apartment bounds
    if (bounds && controlsRef.current) {
      const { halfW, halfD, minY, maxY } = bounds;
      const target = controlsRef.current.target;
      target.x = Math.max(-halfW, Math.min(halfW, target.x));
      target.z = Math.max(-halfD, Math.min(halfD, target.z));
      target.y = Math.max(minY, Math.min(maxY, target.y));

      camera.position.x = Math.max(-halfW * 1.1, Math.min(halfW * 1.1, camera.position.x));
      camera.position.z = Math.max(-halfD * 1.1, Math.min(halfD * 1.1, camera.position.z));
      camera.position.y = Math.max(minY, Math.min(maxY + 2, camera.position.y));
    }
  });

  return (
    <OrbitControls
      ref={controlsRef}
      makeDefault
      enableDamping
      dampingFactor={0.06}
      maxPolarAngle={Math.PI / 2.05}
      minDistance={1}
      maxDistance={12}
    />
  );
}
