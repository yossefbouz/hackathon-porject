import { MapContainer, TileLayer, CircleMarker, Tooltip } from "react-leaflet";
import type { RiskRecord } from "../utils/riskRecord";
import { riskToColor, riskToRadius } from "../utils/riskRecord";

interface RiskMapProps {
  records: RiskRecord[];
}

const CYPRUS_BOUNDS: [[number, number], [number, number]] = [
  [34.46, 32.2],
  [35.77, 34.65],
];

function RiskMap({ records }: RiskMapProps) {
  return (
    <MapContainer
      center={[35.1264, 33.4299]}
      zoom={8}
      scrollWheelZoom={false}
      style={{ height: "100%", width: "100%" }}
      bounds={CYPRUS_BOUNDS}
      maxBoundsViscosity={1}
      minZoom={7}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      {records.map((record) => (
        <CircleMarker
          key={record.id}
          center={[record.latitude, record.longitude]}
          pathOptions={{
            color: riskToColor(record.riskIndex),
            fillColor: riskToColor(record.riskIndex),
            fillOpacity: 0.8,
            weight: 1,
          }}
          radius={riskToRadius(record.riskIndex)}
        >
          <Tooltip direction="top" offset={[0, -8]} opacity={1} permanent={false}>
            <div>
              <strong>{record.name}</strong>
              <div>Risk index: {record.riskIndex.toFixed(2)}</div>
              <div>Wind: {record.windSpeedKmh.toFixed(1)} km/h</div>
              <div>Tree cover: {record.treeCoverPct.toFixed(0)}%</div>
              <div>Vegetation: {record.vegetationType}</div>
            </div>
          </Tooltip>
        </CircleMarker>
      ))}
    </MapContainer>
  );
}

export default RiskMap;
