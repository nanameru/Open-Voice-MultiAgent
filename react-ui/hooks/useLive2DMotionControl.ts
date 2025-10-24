'use client';

import { useDataChannel } from '@livekit/components-react';
import { useRef, useCallback } from 'react';
import type { Live2DModelWrapper } from '@/lib/live2d/Live2DModelWrapper';

/**
 * Live2Dキャラクターのモーションと表情をAIエージェントから制御するフック
 * 
 * バックエンドのLiveKit Agentから送信されるData Channelメッセージを受信し、
 * Live2Dモデルのモーション再生や表情変更を実行します。
 * 
 * @param modelRef - Live2DModelWrapper のリファレンス
 */
export function useLive2DMotionControl(
  modelRef: React.MutableRefObject<Live2DModelWrapper | null>
) {
  const lastProcessedTimestamp = useRef<number>(0);

  const handleMotionMessage = useCallback((data: any) => {
    // メッセージタイプを確認
    if (data.type !== 'live2d_motion') {
      return;
    }

    // モデルが初期化されているか確認
    if (!modelRef.current) {
      console.warn('[useLive2DMotionControl] Model not ready');
      return;
    }

    // 重複メッセージの防止（タイムスタンプベース）
    const now = Date.now();
    if (now - lastProcessedTimestamp.current < 100) {
      // 100ms以内の重複メッセージは無視
      return;
    }
    lastProcessedTimestamp.current = now;

    // アクションに応じて処理を分岐
    switch (data.action) {
      case 'play':
        // モーション再生（グループからランダム）
        console.log(`[Live2D] Playing motion: ${data.motion} (priority: ${data.priority || 3})`);
        modelRef.current.startRandomMotion(
          data.motion,  // "Idle", "TapBody" など
          data.priority || 3
        );
        break;

      case 'play_file':
        // モーション再生（ファイル名を直接指定）
        console.log(`[Live2D] Playing motion file: ${data.motion_file} (priority: ${data.priority || 5})`);
        modelRef.current.playMotionByFile(
          data.motion_file,  // "haru_g_m01" など
          data.priority || 5
        );
        break;

      case 'expression':
        // 表情変更
        console.log(`[Live2D] Setting expression: ${data.name}`);
        modelRef.current.setExpression(data.name);
        break;

      default:
        console.warn(`[Live2D] Unknown action: ${data.action}`);
    }
  }, [modelRef]);

  // LiveKitのData Channelからメッセージを受信
  useDataChannel((message) => {
    console.log('[useLive2DMotionControl] Received data channel message:', {
      payload: message.payload,
      from: message.from?.identity,
    });
    
    try {
      // バイナリデータをテキストにデコード
      const decoder = new TextDecoder();
      const text = decoder.decode(message.payload);
      console.log('[useLive2DMotionControl] Decoded text:', text);
      
      // JSONをパース
      const data = JSON.parse(text);
      console.log('[useLive2DMotionControl] Parsed JSON:', data);
      
      // モーションメッセージを処理
      handleMotionMessage(data);
    } catch (error) {
      // パースエラーは無視（他のData Channelメッセージの可能性）
      console.debug('[useLive2DMotionControl] Failed to parse message:', error);
    }
  });
}

