import { useEffect, useMemo, useState } from "react";
import Papa from "papaparse";
import {
  toRiskRecord,
  type RiskRecord,
  type RawRiskRecord,
} from "../utils/riskRecord";

interface UseRiskDatasetResult {
  records: RiskRecord[];
  loading: boolean;
  error: Error | null;
}

export function useRiskDataset(url: string): UseRiskDatasetResult {
  const [records, setRecords] = useState<RiskRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let cancelled = false;

    setLoading(true);
    setError(null);

    Papa.parse<RawRiskRecord>(url, {
      download: true,
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        if (cancelled) {
          return;
        }

        const parsedRecords = results.data
          .map((entry) => toRiskRecord(entry))
          .filter((record): record is RiskRecord => record !== null);

        setRecords(parsedRecords);
        setLoading(false);
      },
      error: (err) => {
        if (cancelled) {
          return;
        }
        setError(err as Error);
        setLoading(false);
      },
    });

    return () => {
      cancelled = true;
    };
  }, [url]);

  const stableRecords = useMemo(() => records, [records]);

  return { records: stableRecords, loading, error };
}
