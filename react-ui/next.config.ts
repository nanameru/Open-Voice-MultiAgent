import type { NextConfig } from 'next';
import path from 'path';

const nextConfig: NextConfig = {
  /* config options here */
  webpack: (config) => {
    // Live2D Frameworkのエイリアス設定
    config.resolve.alias['@framework'] = path.resolve(__dirname, 'lib/live2d/framework/src');
    
    // WebAssembly対応（Cubism Coreが使用する場合）
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
    };
    
    return config;
  },
};

export default nextConfig;
