// View settings that outlive the window (filters, grouping, chart axes), per key.
import { useCallback, useState } from "react";

export function useStored<T>(key: string, initial: T): [T, (value: T) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      const saved = localStorage.getItem(`kiseki:${key}`);
      return saved == null ? initial : (JSON.parse(saved) as T);
    } catch {
      return initial;
    }
  });
  const store = useCallback(
    (next: T) => {
      setValue(next);
      localStorage.setItem(`kiseki:${key}`, JSON.stringify(next));
    },
    [key],
  );
  return [value, store];
}
