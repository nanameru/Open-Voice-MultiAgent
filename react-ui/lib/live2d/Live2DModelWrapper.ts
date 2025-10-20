/**
 * Live2D Cubism SDKのラッパークラス
 * 複雑な低レベルAPIを隠蔽し、React向けの高レベルインターフェースを提供
 */

import { CubismFramework, Option } from '@framework/live2dcubismframework';
import { CubismUserModel } from '@framework/model/cubismusermodel';
import { CubismModelSettingJson } from '@framework/cubismmodelsettingjson';
import { ICubismModelSetting } from '@framework/icubismmodelsetting';
import type { ExpressionParams } from './types';

// Live2DCubismCoreへの参照
declare const Live2DCubismCore: any;

/**
 * Live2Dモデルラッパークラス
 */
export class Live2DModelWrapper {
  private _model: CubismUserModel | null = null;
  private _canvas: HTMLCanvasElement | null = null;
  private _gl: WebGLRenderingContext | null = null;
  private _animationId: number | null = null;
  private _lastUpdateTime: number = 0;
  private _isInitialized: boolean = false;
  private _initPromise: Promise<void> | null = null;

  constructor() {
    // 初期化は遅延実行（loadModel時に実行）
  }

  /**
   * Cubism Frameworkの初期化（遅延実行・1回のみ）
   */
  private async ensureInitialized(): Promise<void> {
    if (this._isInitialized) return;

    // 既に初期化中の場合は、その完了を待つ
    if (this._initPromise) {
      await this._initPromise;
      return;
    }

    // 初期化開始
    this._initPromise = this.initializeCubism();
    await this._initPromise;
  }

  /**
   * Cubism Frameworkの初期化
   */
  private async initializeCubism(): Promise<void> {
    // Live2DCubismCoreがロードされるまで待機
    await this.waitForCubismCore();

    try {
      // Cubism Frameworkの初期化オプション
      const cubismOption: Option = {
        logFunction: (message: string) => console.log('[Live2D]', message),
        loggingLevel: 0, // 本番環境ではログを抑制
      };

      CubismFramework.startUp(cubismOption);
      CubismFramework.initialize();

      this._isInitialized = true;
      console.log('[Live2DModelWrapper] Cubism Framework initialized');
    } catch (error) {
      console.error('[Live2DModelWrapper] Failed to initialize Cubism Framework:', error);
      this._initPromise = null; // エラー時はリトライ可能にする
      throw error;
    }
  }

  /**
   * Live2DCubismCoreがロードされるまで待機
   */
  private async waitForCubismCore(): Promise<void> {
    const maxAttempts = 50; // 最大5秒待機
    let attempts = 0;

    return new Promise((resolve, reject) => {
      const checkCore = () => {
        if (typeof Live2DCubismCore !== 'undefined') {
          console.log('[Live2DModelWrapper] Live2DCubismCore loaded');
          resolve();
          return;
        }

        attempts++;
        if (attempts >= maxAttempts) {
          reject(new Error('Live2DCubismCore failed to load'));
          return;
        }

        setTimeout(checkCore, 100);
      };

      checkCore();
    });
  }

  /**
   * モデルの読み込み
   * @param modelPath - model3.jsonのパス
   */
  async loadModel(modelPath: string): Promise<void> {
    // Cubism Frameworkの初期化を確認
    await this.ensureInitialized();

    try {
      // JSONファイルを取得
      const response = await fetch(modelPath);
      if (!response.ok) {
        throw new Error(`Failed to load model: ${response.statusText}`);
      }

      const modelJsonText = await response.text();
      const encoder = new TextEncoder();
      const modelJsonBuffer = encoder.encode(modelJsonText).buffer;
      const setting = new CubismModelSettingJson(modelJsonBuffer, modelJsonBuffer.byteLength);

      // モデルファイルのベースパス
      const basePath = modelPath.substring(0, modelPath.lastIndexOf('/') + 1);

      // .moc3ファイルを読み込み
      const mocFileName = setting.getModelFileName();
      const mocPath = basePath + mocFileName;
      const mocResponse = await fetch(mocPath);
      const mocArrayBuffer = await mocResponse.arrayBuffer();

      // モデルを作成（簡略版 - 実際にはCubismUserModelを継承したクラスが必要）
      console.log('[Live2DModelWrapper] Model loaded:', modelPath);
      
      // TODO: 本格的な実装では、CubismUserModelを継承したカスタムクラスでモデルを管理
      // 現在は基本的なログ出力のみ
      
    } catch (error) {
      console.error('[Live2DModelWrapper] Failed to load model:', error);
      throw error;
    }
  }

  /**
   * レンダリング開始
   * @param canvas - 描画先のCanvas要素
   */
  async startRendering(canvas: HTMLCanvasElement): Promise<void> {
    if (!canvas) {
      console.error('[Live2DModelWrapper] Canvas is null');
      return;
    }

    // Cubism Frameworkの初期化を確認
    await this.ensureInitialized();

    this._canvas = canvas;

    // WebGLコンテキストを取得
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (!gl) {
      console.error('[Live2DModelWrapper] Failed to get WebGL context');
      return;
    }

    this._gl = gl as WebGLRenderingContext;

    // レンダリングループ開始
    this._lastUpdateTime = Date.now();
    this.renderLoop();

    console.log('[Live2DModelWrapper] Rendering started');
  }

  /**
   * レンダリングループ
   */
  private renderLoop = (): void => {
    if (!this._canvas || !this._gl) return;

    // 時間更新
    const now = Date.now();
    const deltaTime = (now - this._lastUpdateTime) / 1000.0;
    this._lastUpdateTime = now;

    // テスト描画: 赤い画面でWebGLが動作していることを確認
    this._gl.clearColor(1.0, 0.0, 0.0, 1.0); // 赤色
    this._gl.clear(this._gl.COLOR_BUFFER_BIT);

    // モデル更新と描画
    if (this._model) {
      // TODO: モデル更新処理を実装
      // this._model.update();
      // TODO: 描画処理を実装
    }

    // デバッグ: 最初の数フレームだけログ出力
    if (deltaTime > 0 && Date.now() - this._lastUpdateTime < 1000) {
      console.log('[Live2DModelWrapper] Rendering frame, deltaTime:', deltaTime.toFixed(3));
    }

    // 次のフレームをリクエスト
    this._animationId = requestAnimationFrame(this.renderLoop);
  };

  /**
   * 表情設定
   * @param emotion - 感情文字列 ('happy', 'sad', 'neutral'等)
   */
  setExpression(emotion: string): void {
    if (!this._model) {
      console.warn('[Live2DModelWrapper] Model not loaded');
      return;
    }

    console.log('[Live2DModelWrapper] Set expression:', emotion);
    
    // TODO: 表情モーション再生の実装
    // this._model.setExpression(emotion);
  }

  /**
   * モーション再生
   * @param group - モーショングループ名
   * @param id - モーションID
   * @param priority - 優先度
   */
  playMotion(group: string, id: string, priority: number = 2): void {
    if (!this._model) {
      console.warn('[Live2DModelWrapper] Model not loaded');
      return;
    }

    console.log('[Live2DModelWrapper] Play motion:', group, id);
    
    // TODO: モーション再生の実装
    // this._model.startMotion(group, id, priority);
  }

  /**
   * リップシンク設定
   * @param value - 口の開き具合 (0.0-1.0)
   */
  setLipSync(value: number): void {
    if (!this._model) return;

    // TODO: リップシンクパラメータの設定
    // this._model.setParameterValueById('ParamMouthOpenY', value);
  }

  /**
   * パラメータを直接設定
   * @param params - 表情パラメータ
   */
  setParameters(params: Partial<ExpressionParams>): void {
    if (!this._model) return;

    console.log('[Live2DModelWrapper] Set parameters:', params);
    
    // TODO: パラメータ設定の実装
  }

  /**
   * リソース解放
   */
  destroy(): void {
    console.log('[Live2DModelWrapper] Destroying...');

    // レンダリング停止
    if (this._animationId !== null) {
      cancelAnimationFrame(this._animationId);
      this._animationId = null;
    }

    // モデル解放
    if (this._model) {
      this._model.release();
      this._model = null;
    }

    // WebGLコンテキスト解放
    this._gl = null;
    this._canvas = null;

    console.log('[Live2DModelWrapper] Destroyed');
  }
}

