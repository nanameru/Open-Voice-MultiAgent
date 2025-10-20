/**
 * Live2D統合のためのTypeScript型定義
 */

import type { AgentState } from '@livekit/components-react';

/**
 * Visemeデータ（リップシンク用）
 */
export interface VisemeData {
  /** 秒単位のタイムスタンプ */
  time: number;
  /** Viseme種別 ('aa', 'ih', 'ou', 'eh'等) */
  viseme: string;
  /** 0.0-1.0の強度 */
  value: number;
}

/**
 * Live2D制御コマンド（WebSocket経由で送信）
 */
export interface Live2DControlCommand {
  /** コマンドタイプ */
  type: 'expression' | 'motion' | 'lipsync';
  /** 感情（expression用） */
  emotion?: string;
  /** モーション情報（motion用） */
  motion?: {
    group: string;
    id: string;
    priority?: number;
  };
  /** Visemeデータ（lipsync用） */
  visemeData?: VisemeData[];
}

/**
 * 表情パラメータ
 */
export interface ExpressionParams {
  /** 口の開き具合 Y軸 */
  mouthOpenY: number;
  /** 左目の開き具合 */
  eyeLOpen: number;
  /** 右目の開き具合 */
  eyeROpen: number;
  /** 左眉毛の位置 Y軸 */
  eyebrowLY: number;
  /** 右眉毛の位置 Y軸 */
  eyebrowRY: number;
}

/**
 * Live2Dモデル設定
 */
export interface Live2DModelConfig {
  /** モデルファイルのパス */
  modelPath: string;
  /** Canvas幅 */
  width?: number;
  /** Canvas高さ */
  height?: number;
  /** 自動アイドルモーション再生 */
  autoPlay?: boolean;
}

/**
 * Live2DBackgroundコンポーネントのProps
 */
export interface Live2DBackgroundProps {
  /** エージェント状態 */
  agentState: AgentState;
  /** モデル設定 */
  config?: Partial<Live2DModelConfig>;
  /** クラス名 */
  className?: string;
}

/**
 * 感情からLive2Dパラメータへのマッピング
 */
export type EmotionMapping = {
  [emotion: string]: ExpressionParams;
};

