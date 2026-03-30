import { useGLTF, Environment } from '@react-three/drei';
import { Suspense, useEffect, useRef, useCallback } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import { Box3, Vector3, Color, Mesh, Material, MeshStandardMaterial, Object3D, PointLight } from 'three';

export interface SceneObject {
  name: string;
  cleanName: string;
  worldPosition: [number, number, number];
}

interface Props {
  onBoundsReady?: (center: [number, number, number], size: [number, number, number]) => void;
  onObjectsDiscovered?: (objects: SceneObject[]) => void;
  searchTarget?: string | null;
  onObjectFound?: (position: [number, number, number], name: string) => void;
}

function ApartmentModel({ onBoundsReady, onObjectsDiscovered, searchTarget, onObjectFound }: Props) {
  const { scene } = useGLTF('/models/apartment.glb');
  const { camera } = useThree();
  const objectsRef = useRef<{ obj: SceneObject; mesh: Object3D }[]>([]);
  const highlightLightRef = useRef<PointLight | null>(null);
  const highlightedMeshes = useRef<{ mesh: Mesh; origMaterials: Material | Material[] }[]>([]);

  // Parse model on load — discover all named meshes
  useEffect(() => {
    const box = new Box3().setFromObject(scene);
    const center = box.getCenter(new Vector3());
    const size = box.getSize(new Vector3());

    scene.position.x -= center.x;
    scene.position.z -= center.z;

    const camY = center.y + size.y * 0.7;
    const camZ = size.z * 0.4;
    camera.position.set(0, camY, camZ);
    camera.lookAt(0, center.y, 0);

    onBoundsReady?.([0, center.y, 0], [size.x, size.y, size.z]);

    // Traverse and collect all named meshes
    const discovered: { obj: SceneObject; mesh: Object3D }[] = [];
    const seen = new Set<string>();
    scene.traverse((child) => {
      if ((child as Mesh).isMesh && child.name && !seen.has(child.name)) {
        seen.add(child.name);
        const pos = new Vector3();
        child.getWorldPosition(pos);
        // Clean the name: "Kitchen_Cup_001" → "cup"
        const cleanName = child.name
          .replace(/[_.\d]+/g, ' ')
          .replace(/\b(mesh|object|group|node|geo|mat|material)\b/gi, '')
          .trim()
          .toLowerCase();
        if (cleanName.length > 1) {
          discovered.push({
            obj: {
              name: child.name,
              cleanName,
              worldPosition: [pos.x, pos.y, pos.z],
            },
            mesh: child,
          });
        }
      }
    });

    objectsRef.current = discovered;
    console.log(
      `Discovered ${discovered.length} meshes:`,
      discovered.map((d) => `${d.obj.name} → "${d.obj.cleanName}" @ [${d.obj.worldPosition.map((v) => v.toFixed(1)).join(',')}]`),
    );
    onObjectsDiscovered?.(discovered.map((d) => d.obj));
  }, [scene, camera, onBoundsReady, onObjectsDiscovered]);

  // Handle search — find and highlight matching mesh
  useEffect(() => {
    // Clear previous highlights
    for (const h of highlightedMeshes.current) {
      (h.mesh as Mesh).material = h.origMaterials as Material;
    }
    highlightedMeshes.current = [];

    if (!searchTarget) return;

    const q = searchTarget.toLowerCase().replace('find the ', '').replace('find ', '').trim();
    if (!q) return;

    // Fuzzy match: find meshes whose name contains the query (or vice versa)
    const matches = objectsRef.current.filter((entry) => {
      const n = entry.obj.cleanName;
      const raw = entry.obj.name.toLowerCase();
      return n.includes(q) || q.includes(n) || raw.includes(q) || q.split(' ').some((w) => w.length > 2 && (n.includes(w) || raw.includes(w)));
    });

    if (matches.length === 0) {
      console.log(`No mesh found matching "${q}"`);
      return;
    }

    // Pick the first match
    const best = matches[0];
    console.log(`Found "${q}" → mesh "${best.obj.name}" at [${best.obj.worldPosition.join(',')}]`);

    // Highlight all matching meshes
    for (const match of matches.slice(0, 5)) {
      const mesh = match.mesh as Mesh;
      if (!mesh.isMesh) continue;

      // Clone and modify material to glow
      const origMat = mesh.material;
      highlightedMeshes.current.push({ mesh, origMaterials: origMat });

      if (Array.isArray(origMat)) {
        mesh.material = origMat.map((m) => {
          const clone = (m as MeshStandardMaterial).clone();
          clone.emissive = new Color('#3fb950');
          clone.emissiveIntensity = 1.5;
          return clone;
        });
      } else {
        const clone = (origMat as MeshStandardMaterial).clone();
        clone.emissive = new Color('#3fb950');
        clone.emissiveIntensity = 1.5;
        mesh.material = clone;
      }
    }

    // Notify parent with exact 3D position
    onObjectFound?.(best.obj.worldPosition, best.obj.cleanName);
  }, [searchTarget, onObjectFound]);

  // Pulsing highlight animation
  useFrame(({ clock }) => {
    const t = clock.elapsedTime;
    const intensity = 1.0 + Math.sin(t * 3) * 0.5;
    for (const h of highlightedMeshes.current) {
      const mat = (h.mesh as Mesh).material;
      if (Array.isArray(mat)) {
        mat.forEach((m) => {
          (m as MeshStandardMaterial).emissiveIntensity = intensity;
        });
      } else {
        (mat as MeshStandardMaterial).emissiveIntensity = intensity;
      }
    }
  });

  return <primitive object={scene} />;
}

export default function ApartmentScene(props: Props) {
  return (
    <Suspense fallback={null}>
      <ApartmentModel {...props} />
      <Environment preset="apartment" background={false} environmentIntensity={0.4} />
      <ambientLight intensity={0.6} color="#aabbdd" />
      <directionalLight
        position={[5, 12, 8]}
        intensity={0.8}
        color="#ffeedd"
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />
      <pointLight position={[0, 1, 0]} intensity={0.3} color="#ffffff" distance={30} />
    </Suspense>
  );
}
