export interface CameraTarget {
  position: [number, number, number];
  lookAt: [number, number, number];
}

// Model bounds — updated dynamically when the GLB loads
let modelCenter: [number, number, number] = [0, 1.5, 0];
let modelSize: [number, number, number] = [12, 3, 9];

export function setModelBounds(
  center: [number, number, number],
  size: [number, number, number],
) {
  modelCenter = center;
  modelSize = size;
}

/**
 * Map grid coords (0-3 x, 0-2 y) to a 3D position INSIDE the model.
 * Divides the model's footprint into a 4x3 grid.
 */
export function gridTo3D(gx: number, gy: number): [number, number, number] {
  const halfW = modelSize[0] / 2;
  const halfD = modelSize[2] / 2;
  const cellW = modelSize[0] / 4;
  const cellD = modelSize[2] / 3;
  const x = -halfW + cellW * gx + cellW / 2;
  const z = -halfD + cellD * gy + cellD / 2;
  return [x, modelCenter[1] + 0.3, z];
}

/**
 * Camera position to look at a specific room from inside the apartment.
 */
export function getCameraForRoom(gx: number, gy: number): CameraTarget {
  const [rx, ry, rz] = gridTo3D(gx, gy);
  return {
    position: [rx + modelSize[0] * 0.08, ry + 1.5, rz + modelSize[2] * 0.12],
    lookAt: [rx, ry - 0.5, rz],
  };
}

/**
 * Overview camera: above center, looking down into the apartment.
 */
export function getOverviewCamera(): CameraTarget {
  return {
    position: [0, modelCenter[1] + modelSize[1] * 0.7, modelSize[2] * 0.4],
    lookAt: [0, modelCenter[1], 0],
  };
}
