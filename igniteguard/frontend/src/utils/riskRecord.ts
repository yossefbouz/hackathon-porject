export interface RawRiskRecord {
  id?: string;
  name?: string;
  latitude?: string;
  longitude?: string;
  risk_index?: string;
  wind_speed_kmh?: string;
  tree_cover_pct?: string;
  vegetation_type?: string;
}

export interface RiskRecord {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  riskIndex: number;
  windSpeedKmh: number;
  treeCoverPct: number;
  vegetationType: string;
}

export function toRiskRecord(value: RawRiskRecord | null): RiskRecord | null {
  if (!value) {
    return null;
  }

  const latitude = Number(value.latitude);
  const longitude = Number(value.longitude);
  const rawRisk = Number(value.risk_index);
  const windSpeed = Number(value.wind_speed_kmh);
  const treeCover = Number(value.tree_cover_pct);

  if (
    Number.isNaN(latitude) ||
    Number.isNaN(longitude) ||
    Number.isNaN(rawRisk) ||
    Number.isNaN(windSpeed) ||
    Number.isNaN(treeCover)
  ) {
    return null;
  }

  const normalisedRisk = rawRisk > 1 ? rawRisk / 100 : rawRisk;

  return {
    id: value.id ?? `${value.latitude}-${value.longitude}`,
    name: value.name ?? "Unknown site",
    latitude,
    longitude,
    riskIndex: clamp(normalisedRisk),
    windSpeedKmh: windSpeed,
    treeCoverPct: treeCover,
    vegetationType: value.vegetation_type ?? "Unknown",
  };
}

export function riskToColor(riskIndex: number): string {
  const clamped = clamp(riskIndex);
  const hue = (1 - clamped) * 120;
  return `hsl(${hue.toFixed(0)}, 80%, 45%)`;
}

export function riskToRadius(riskIndex: number): number {
  const base = 12;
  const scale = 18;
  return base + clamp(riskIndex) * scale;
}

function clamp(value: number): number {
  return Math.max(0, Math.min(1, value));
}
