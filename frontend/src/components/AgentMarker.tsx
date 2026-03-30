import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Mesh } from 'three';

interface Props {
  position: [number, number, number];
}

export default function AgentMarker({ position }: Props) {
  const meshRef = useRef<Mesh>(null);

  useFrame(({ clock }) => {
    if (meshRef.current) {
      meshRef.current.position.y = position[1] + Math.sin(clock.elapsedTime * 3) * 0.08;
    }
  });

  return (
    <group position={position}>
      {/* Outer glow */}
      <mesh>
        <sphereGeometry args={[0.45, 16, 16]} />
        <meshBasicMaterial color="#f0c040" transparent opacity={0.12} />
      </mesh>
      {/* Inner orb */}
      <mesh ref={meshRef}>
        <sphereGeometry args={[0.22, 24, 24]} />
        <meshStandardMaterial
          color="#f0c040"
          emissive="#f0c040"
          emissiveIntensity={1.0}
          roughness={0.2}
          metalness={0.5}
        />
      </mesh>
    </group>
  );
}
