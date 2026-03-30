import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { DoubleSide, Mesh } from 'three';

interface Props {
  position: [number, number, number];
  label: string;
  confidence: number;
}

export default function DetectionMarker({ position, label, confidence }: Props) {
  const ringRef = useRef<Mesh>(null);
  const beamRef = useRef<Mesh>(null);
  const color = confidence >= 0.8 ? '#3fb950' : confidence >= 0.5 ? '#d29922' : '#f85149';

  useFrame(({ clock }) => {
    if (ringRef.current) {
      ringRef.current.rotation.z = clock.elapsedTime * 0.5;
      const scale = 1 + Math.sin(clock.elapsedTime * 2) * 0.1;
      ringRef.current.scale.set(scale, scale, scale);
    }
    if (beamRef.current) {
      const opacity = 0.15 + Math.sin(clock.elapsedTime * 3) * 0.1;
      (beamRef.current.material as any).opacity = opacity;
    }
  });

  return (
    <group position={position}>
      {/* Pulsing ring on the floor */}
      <mesh ref={ringRef} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.05, 0]}>
        <ringGeometry args={[0.4, 0.6, 32]} />
        <meshBasicMaterial color={color} transparent opacity={0.7} side={DoubleSide} />
      </mesh>

      {/* Outer ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.04, 0]}>
        <ringGeometry args={[0.7, 0.75, 32]} />
        <meshBasicMaterial color={color} transparent opacity={0.3} side={DoubleSide} />
      </mesh>

      {/* Vertical beam of light */}
      <mesh ref={beamRef} position={[0, 1, 0]}>
        <cylinderGeometry args={[0.03, 0.03, 2, 8]} />
        <meshBasicMaterial color={color} transparent opacity={0.2} />
      </mesh>

      {/* Top dot */}
      <mesh position={[0, 2.1, 0]}>
        <sphereGeometry args={[0.08, 12, 12]} />
        <meshBasicMaterial color={color} />
      </mesh>
    </group>
  );
}
