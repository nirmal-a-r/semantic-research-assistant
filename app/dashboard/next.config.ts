import type { NextConfig } from "next";

/**
 * Two build modes:
 *
 *  npm run dev / npm run build      → normal Next.js app (talks to the FastAPI
 *                                     backend on localhost:8000 when it is up)
 *
 *  BASE_PATH=/semantic-research-assistant npm run build:static
 *                                   → fully static export in `out/`, served by
 *                                     GitHub Pages with no server of any kind.
 *                                     Every metric and demo answer is read from
 *                                     public/data/project_data.json.
 */
const isStatic = process.env.STATIC_EXPORT === "true";
const basePath = process.env.BASE_PATH ?? "";

const nextConfig: NextConfig = {
  reactCompiler: true,
  ...(isStatic
    ? {
        output: "export" as const,
        basePath,
        assetPrefix: basePath || undefined,
        trailingSlash: true,
        images: { unoptimized: true },
      }
    : {}),
  env: {
    // fetch() and plain <a href> are not rewritten by basePath — see lib/data.ts
    NEXT_PUBLIC_BASE_PATH: isStatic ? basePath : "",
  },
};

export default nextConfig;
