"use client";
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { getProjectData, type ProjectData } from "@/lib/data";

interface ProjectDataContextType {
  data: ProjectData | null;
  loading: boolean;
  error: string | null;
}

const ProjectDataContext = createContext<ProjectDataContextType>({
  data: null,
  loading: true,
  error: null,
});

export function ProjectDataProvider({ children }: { children: ReactNode }) {
  const [data, setData]     = useState<ProjectData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]   = useState<string | null>(null);

  useEffect(() => {
    getProjectData()
      .then(setData)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  return (
    <ProjectDataContext.Provider value={{ data, loading, error }}>
      {children}
    </ProjectDataContext.Provider>
  );
}

export function useProjectData() {
  return useContext(ProjectDataContext);
}
