import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  reactCompiler: true,
  async rewrites() {
    return [
      {
        source: '/api/test',
        destination: 'http://127.0.0.1:8000/test',
      },
    ]
  },
};

export default nextConfig;
