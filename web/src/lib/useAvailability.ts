import { useEffect, useState } from "react";
import { checkHealth } from "./api";

export function useAvailability() {
  const [available, setAvailable] = useState<boolean | null>(null);
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      let ready = false;
      try { ready = (await checkHealth()).status === "ok"; } catch { /* unavailable */ }
      if (active) {
        setAvailable(ready);
        timer = setTimeout(refresh, 5000);
      }
    }
    void refresh();
    return () => { active = false; clearTimeout(timer); };
  }, []);
  return available;
}
