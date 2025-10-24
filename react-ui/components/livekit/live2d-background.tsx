'use client';

import { useEffect, useRef } from 'react';
import type { AgentState } from '@livekit/components-react';
import { Live2DModelWrapper } from '@/lib/live2d/Live2DModelWrapper';
import { useLive2DMotionControl } from '@/hooks/useLive2DMotionControl';
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

  // AIエージェントからのモーション制御を有効化
  useLive2DMotionControl(modelRef);

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
        // F01: 通常の表情
        modelRef.current.setExpression('F01');
        break;
      case 'thinking':
        // F03: 考え中の表情
        modelRef.current.setExpression('F03');
        console.log('[Live2DBackground] Agent is thinking...');
        break;
      case 'speaking':
        // F02: 笑顔
        modelRef.current.setExpression('F02');
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
        'flex items-center justify-center',
        'pointer-events-none',
        className
      )}
    >
      {/* Live2Dキャラクターを画面中央に配置 */}
      <canvas
        ref={canvasRef}
        width={800}
        height={1200}
        className="max-h-[80vh]"
        style={{ background: 'transparent' }}
      />
    </div>
  );
}

