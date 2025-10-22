'use client';

import { useEffect, useRef } from 'react';
import type { AgentState } from '@livekit/components-react';
import { Live2DModelWrapper } from '@/lib/live2d/Live2DModelWrapper';
import { cn } from '@/lib/utils';

interface Live2DBackgroundProps {
  agentState: AgentState;
  className?: string;
}

/**
 * Live2Dキャラクターを背景レイヤーとして表示するコンポーネント
 */
export function Live2DBackground({ agentState, className }: Live2DBackgroundProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modelRef = useRef<Live2DModelWrapper | null>(null);

  // モデル初期化
  useEffect(() => {
    const initModel = async () => {
      if (!canvasRef.current) {
        console.warn('[Live2DBackground] Canvas ref is null');
        return;
      }

      try {
        console.log('[Live2DBackground] Initializing Live2D model...');
        
        const model = new Live2DModelWrapper();
        await model.loadModel('/live2d/models/Haru/Haru.model3.json');
        await model.startRendering(canvasRef.current);
        
        modelRef.current = model;
        console.log('[Live2DBackground] Live2D model initialized');
      } catch (error) {
        console.error('[Live2DBackground] Failed to initialize model:', error);
      }
    };

    initModel();

    // クリーンアップ
    return () => {
      if (modelRef.current) {
        console.log('[Live2DBackground] Cleaning up Live2D model');
        modelRef.current.destroy();
        modelRef.current = null;
      }
    };
  }, []);

  // エージェント状態に応じた表情変更
  useEffect(() => {
    if (!modelRef.current) return;

    switch (agentState) {
      case 'listening':
        modelRef.current.setExpression('neutral');
        break;
      case 'thinking':
        modelRef.current.setExpression('neutral');
        console.log('[Live2DBackground] Agent is thinking...');
        break;
      case 'speaking':
        modelRef.current.setExpression('happy');
        console.log('[Live2DBackground] Agent is speaking!');
        break;
      default:
        break;
    }
  }, [agentState]);

  return (
    <div
      className={cn(
        'fixed inset-0 z-10',
        'flex items-end justify-center',
        'pointer-events-none',
        className
      )}
    >
      {/* Live2Dキャラクターを画面下部に配置 */}
      <canvas
        ref={canvasRef}
        width={400}
        height={800}
        className="max-h-[calc(100vh-200px)]"
      />
    </div>
  );
}

