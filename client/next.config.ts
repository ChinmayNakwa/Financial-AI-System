import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'export', // This is the key change
  images: {
    unoptimized: true, // Required for static export
  },
};

export default nextConfig;