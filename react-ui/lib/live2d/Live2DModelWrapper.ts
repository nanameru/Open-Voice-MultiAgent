/**
 * Live2DModelWrapper
 * Live2Dモデルの読み込み、更新、描画を管理するラッパークラス
 */

import { CubismFramework, Option } from '@framework/live2dcubismframework';
import { CubismMatrix44 } from '@framework/math/cubismmatrix44';
import { SimpleLive2DModel } from './SimpleLive2DModel';

export class Live2DModelWrapper {
  private _model: SimpleLive2DModel | null = null;
  private _gl: WebGLRenderingContext | null = null;
  private _canvas: HTMLCanvasElement | null = null;
  private _animationId: number | null = null;
  private _lastUpdateTime: number = 0;
  private _isInitialized: boolean = false;
  private _initPromise: Promise<void> | null = null;
  private _projection: CubismMatrix44 | null = null;

  /**
   * Cubismフレームワークを初期化
   */
  private async ensureInitialized(): Promise<void> {
    if (this._isInitialized) return;
    if (this._initPromise) {
      await this._initPromise;
      return;
    }
    this._initPromise = this.initializeCubism();
    await this._initPromise;
  }

  /**
   * Cubism Frameworkの初期化処理
   */
  private async initializeCubism(): Promise<void> {
    // Live2DCubismCoreが読み込まれるまで待機
    await this.waitForCubismCore();

    try {
      const cubismOption: Option = {
        logFunction: (message: string) => {
          // エラーのみログ出力（警告とエラー以外は抑制）
          if (message.includes('[E]') || message.includes('Error')) {
            // NoPremultipliedAlphaエラーは無視（既知の問題）
            if (!message.includes('NoPremultipliedAlpha')) {
              console.error('[Live2D]', message);
            }
          }
        },
        loggingLevel: 2, // LogLevel.Error（エラーのみ）
      };

      CubismFramework.startUp(cubismOption);
      CubismFramework.initialize();

      this._isInitialized = true;
      console.log('[Live2DModelWrapper] Cubism Framework initialized');
    } catch (error) {
      console.error('[Live2DModelWrapper] Failed to initialize Cubism Framework:', error);
      this._initPromise = null;
      throw error;
    }
  }

  /**
   * Live2DCubismCoreが読み込まれるまで待機
   */
  private async waitForCubismCore(): Promise<void> {
    const maxAttempts = 50;
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
   * モデルを読み込む
   * @param modelPath model3.jsonのパス
   */
  async loadModel(modelPath: string): Promise<void> {
    await this.ensureInitialized();

    try {
      // パスを分解
      const lastSlash = modelPath.lastIndexOf('/');
      const dir = modelPath.substring(0, lastSlash + 1);
      const fileName = modelPath.substring(lastSlash + 1);

      console.log('[Live2DModelWrapper] Loading model:', { dir, fileName });

      // モデルを作成
      this._model = new SimpleLive2DModel();

      // モデルをロード
      await this._model.loadAssets(dir, fileName);

      console.log('[Live2DModelWrapper] Model loaded:', modelPath);
    } catch (error) {
      console.error('[Live2DModelWrapper] Failed to load model:', error);
      throw error;
    }
  }

  /**
   * レンダリングを開始
   * @param canvas 描画先のcanvas要素
   */
  async startRendering(canvas: HTMLCanvasElement): Promise<void> {
    await this.ensureInitialized();

    if (!this._model) {
      throw new Error('Model not loaded. Call loadModel() first.');
    }

    this._canvas = canvas;

    // WebGLコンテキストを取得
    const gl = canvas.getContext('webgl', {
      alpha: true,
      premultipliedAlpha: true,
    }) || canvas.getContext('experimental-webgl', {
      alpha: true,
      premultipliedAlpha: true,
    }) as WebGLRenderingContext | null;

    if (!gl) {
      throw new Error('Failed to get WebGL context');
    }

    this._gl = gl;

    console.log('[Live2DModelWrapper] WebGL context created');

    // レンダラーをセットアップ
    this._model.setupRenderer(gl, canvas.width, canvas.height);

    // projection行列を作成
    this._projection = new CubismMatrix44();
    this.updateProjection();

    // モデル行列を設定（画面中央に配置）
    const modelMatrix = this._model.getModelMatrix();
    modelMatrix.setHeight(2.8); // モデルサイズを調整
    modelMatrix.centerX(0.0);
    modelMatrix.setY(-0.2); // 少し下に配置

    // レンダリングループを開始
    this._lastUpdateTime = Date.now();
    this.renderLoop();

    console.log('[Live2DModelWrapper] Rendering started');
  }

  /**
   * projection行列を更新
   */
  private updateProjection(): void {
    if (!this._projection || !this._canvas) return;

    const { width, height } = this._canvas;

    this._projection.loadIdentity();

    if (width > height) {
      // 横長の場合
      const ratio = width / height;
      this._projection.scale(1.0 / ratio, 1.0);
    } else {
      // 縦長の場合
      const ratio = height / width;
      this._projection.scale(1.0, 1.0 / ratio);
    }
  }

  /**
   * レンダリングループ
   */
  private renderLoop = (): void => {
    if (!this._gl || !this._model || !this._projection) {
      console.warn('[Live2DModelWrapper] Rendering components not ready');
      return;
    }

    // 時間更新
    const now = Date.now();
    const deltaTime = (now - this._lastUpdateTime) / 1000.0;
    this._lastUpdateTime = now;

    // 画面クリア（透明）
    this._gl.clearColor(0.0, 0.0, 0.0, 0.0);
    this._gl.clear(this._gl.COLOR_BUFFER_BIT);

    // モデル更新
    this._model.update(deltaTime);

    // モデル描画
    this._model.draw(this._projection);

    // 次のフレームをリクエスト
    this._animationId = requestAnimationFrame(this.renderLoop);
  };

  /**
   * 表情を設定
   */
  setExpression(expressionName: string): void {
    if (this._model) {
      this._model.setExpression(expressionName);
    }
  }

  /**
   * ランダムなモーションを開始
   */
  startRandomMotion(group: string, priority: number = 3): void {
    if (this._model) {
      this._model.startRandomMotion(group, priority);
    }
  }

  /**
   * 指定されたモーションファイルを直接再生
   */
  async playMotionByFile(motionFileName: string, priority: number = 5): Promise<void> {
    if (this._model) {
      await this._model.playMotionByFile(motionFileName, priority);
    }
  }

  /**
   * リソースを解放
   */
  destroy(): void {
    console.log('[Live2DModelWrapper] Destroying...');

    // レンダリングループを停止
    if (this._animationId !== null) {
      cancelAnimationFrame(this._animationId);
      this._animationId = null;
    }

    // モデルを解放
    if (this._model) {
      this._model.release();
      this._model = null;
    }

    // WebGLコンテキストを解放
    this._gl = null;
    this._canvas = null;
    this._projection = null;

    console.log('[Live2DModelWrapper] Destroyed');
  }
}
