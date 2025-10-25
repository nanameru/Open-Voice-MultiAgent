/**
 * SimpleLive2DModel
 * CubismUserModelを継承したLive2Dモデルクラス
 * 表情、モーション、まばたき、呼吸などの機能を実装
 */

import { CubismUserModel } from '@framework/model/cubismusermodel';
import { CubismModelSettingJson } from '@framework/cubismmodelsettingjson';
import { ICubismModelSetting } from '@framework/icubismmodelsetting';
import { CubismPhysics } from '@framework/physics/cubismphysics';
import { CubismMatrix44 } from '@framework/math/cubismmatrix44';
import { CubismEyeBlink } from '@framework/effect/cubismeyeblink';
import { CubismBreath, BreathParameterData } from '@framework/effect/cubismbreath';
import { CubismPose } from '@framework/effect/cubismpose';
import { csmVector } from '@framework/type/csmvector';
import { CubismDefaultParameterId } from '@framework/cubismdefaultparameterid';
import { CubismIdHandle } from '@framework/id/cubismid';
import { CubismFramework } from '@framework/live2dcubismframework';
import { CubismMotion } from '@framework/motion/cubismmotion';
import { CubismExpressionMotion } from '@framework/motion/cubismexpressionmotion';
import { CubismMotionManager } from '@framework/motion/cubismmotionmanager';
import { CubismExpressionMotionManager } from '@framework/motion/cubismexpressionmotionmanager';
import { csmMap } from '@framework/type/csmmap';

enum LoadStep {
  LoadAssets,
  LoadModel,
  LoadExpression,
  LoadPhysics,
  LoadPose,
  SetupEyeBlink,
  SetupBreath,
  LoadMotion,
  CompleteSetup,
}

export class SimpleLive2DModel extends CubismUserModel {
  private _modelHomeDir: string = '';
  private _modelSetting: ICubismModelSetting | null = null;
  private _state: LoadStep = LoadStep.LoadAssets;
  private _userTimeSeconds: number = 0.0;
  private _textures: WebGLTexture[] = [];
  private _gl: WebGLRenderingContext | null = null;
  
  // モーション管理用（アイドルモーション遅延制御）
  private _lastMotionTime: number = 0.0;
  private _idleDelay: number = 3.0; // アイドルモーション再生までの遅延（秒）

  // モーション・表情管理（基底クラスの_motionManagerは使用）
  private _expressions: csmMap<string, CubismExpressionMotion> = new csmMap();
  private _motions: csmMap<string, CubismMotion> = new csmMap();

  // まばたき・リップシンク用パラメータID
  private _eyeBlinkIds: csmVector<CubismIdHandle> = new csmVector();
  private _lipSyncIds: csmVector<CubismIdHandle> = new csmVector();

  // パラメータID
  private _idParamAngleX: CubismIdHandle | null = null;
  private _idParamAngleY: CubismIdHandle | null = null;
  private _idParamAngleZ: CubismIdHandle | null = null;
  private _idParamBodyAngleX: CubismIdHandle | null = null;
  private _idParamEyeBallX: CubismIdHandle | null = null;
  private _idParamEyeBallY: CubismIdHandle | null = null;

  /**
   * model3.jsonが置かれたディレクトリとファイルパスからモデルを生成する
   */
  public async loadAssets(dir: string, fileName: string): Promise<void> {
    this._modelHomeDir = dir;

    console.log('[SimpleLive2DModel] Loading model from:', `${this._modelHomeDir}${fileName}`);

    // model3.jsonを読み込む
    const response = await fetch(`${this._modelHomeDir}${fileName}`);
    const arrayBuffer = await response.arrayBuffer();

    const setting: ICubismModelSetting = new CubismModelSettingJson(
      arrayBuffer,
      arrayBuffer.byteLength
    );

    this._modelSetting = setting;
    this._state = LoadStep.LoadModel;

    await this.setupModel();
  }

  /**
   * モデルをセットアップする
   */
  private async setupModel(): Promise<void> {
    if (!this._modelSetting) {
      console.error('[SimpleLive2DModel] Model setting is null');
      return;
    }

    // .moc3ファイルを読み込み
    const mocFileName = this._modelSetting.getModelFileName();
    const mocResponse = await fetch(`${this._modelHomeDir}${mocFileName}`);
    const mocArrayBuffer = await mocResponse.arrayBuffer();

    console.log('[SimpleLive2DModel] Loading .moc3 file:', mocFileName);

    // 基底クラスのloadModelメソッドを使用してモデルを作成
    this.loadModel(mocArrayBuffer, false);

    if (!this._model) {
      console.error('[SimpleLive2DModel] Failed to create CubismModel');
      return;
    }

    console.log('[SimpleLive2DModel] Model created successfully');

    // パラメータIDを取得
    this._idParamAngleX = CubismFramework.getIdManager().getId(CubismDefaultParameterId.ParamAngleX);
    this._idParamAngleY = CubismFramework.getIdManager().getId(CubismDefaultParameterId.ParamAngleY);
    this._idParamAngleZ = CubismFramework.getIdManager().getId(CubismDefaultParameterId.ParamAngleZ);
    this._idParamBodyAngleX = CubismFramework.getIdManager().getId(CubismDefaultParameterId.ParamBodyAngleX);
    this._idParamEyeBallX = CubismFramework.getIdManager().getId(CubismDefaultParameterId.ParamEyeBallX);
    this._idParamEyeBallY = CubismFramework.getIdManager().getId(CubismDefaultParameterId.ParamEyeBallY);

    // 各種データを順次読み込み
    await this.loadExpressions();
    await this.loadPhysics();
    await this.loadPose();
    this.setupEyeBlink();
    this.setupBreath();
    this.setupEyeBlinkIds();
    this.setupLipSyncIds();
    await this.loadMotions();

    this._state = LoadStep.CompleteSetup;
    console.log('[SimpleLive2DModel] Setup complete');
  }

  /**
   * 表情を読み込む
   */
  private async loadExpressions(): Promise<void> {
    if (!this._modelSetting) return;

    const expressionCount = this._modelSetting.getExpressionCount();
    if (expressionCount === 0) {
      console.log('[SimpleLive2DModel] No expressions');
      return;
    }

    console.log('[SimpleLive2DModel] Loading', expressionCount, 'expressions');

    for (let i = 0; i < expressionCount; i++) {
      const expressionName = this._modelSetting.getExpressionName(i);
      const expressionFileName = this._modelSetting.getExpressionFileName(i);

      try {
        const response = await fetch(`${this._modelHomeDir}${expressionFileName}`);
        const arrayBuffer = await response.arrayBuffer();

        const expression = CubismExpressionMotion.create(arrayBuffer, arrayBuffer.byteLength);
        if (expression) {
          this._expressions.setValue(expressionName, expression);
          console.log('[SimpleLive2DModel] Expression loaded:', expressionName);
        }
      } catch (error) {
        console.warn('[SimpleLive2DModel] Failed to load expression:', expressionName, error);
      }
    }
  }

  /**
   * 物理演算を読み込む
   */
  public async loadPhysics(): Promise<void> {
    if (!this._modelSetting) return;

    const physicsFileName = this._modelSetting.getPhysicsFileName();
    if (physicsFileName === '') {
      console.log('[SimpleLive2DModel] No physics file');
      return;
    }

    try {
      const response = await fetch(`${this._modelHomeDir}${physicsFileName}`);
      const arrayBuffer = await response.arrayBuffer();

      this._physics = CubismPhysics.create(arrayBuffer, arrayBuffer.byteLength);
      console.log('[SimpleLive2DModel] Physics loaded');
    } catch (error) {
      console.warn('[SimpleLive2DModel] Failed to load physics:', error);
    }
  }

  /**
   * ポーズを読み込む
   */
  public async loadPose(): Promise<void> {
    if (!this._modelSetting) return;

    const poseFileName = this._modelSetting.getPoseFileName();
    if (poseFileName === '') {
      console.log('[SimpleLive2DModel] No pose file');
      return;
    }

    try {
      const response = await fetch(`${this._modelHomeDir}${poseFileName}`);
      const arrayBuffer = await response.arrayBuffer();

      this._pose = CubismPose.create(arrayBuffer, arrayBuffer.byteLength);
      console.log('[SimpleLive2DModel] Pose loaded');
    } catch (error) {
      console.warn('[SimpleLive2DModel] Failed to load pose:', error);
    }
  }

  /**
   * まばたきをセットアップ
   */
  private setupEyeBlink(): void {
    if (!this._modelSetting) return;

    if (this._modelSetting.getEyeBlinkParameterCount() > 0) {
      this._eyeBlink = CubismEyeBlink.create(this._modelSetting);
      console.log('[SimpleLive2DModel] EyeBlink setup');
    }
  }

  /**
   * 呼吸をセットアップ
   */
  private setupBreath(): void {
    this._breath = CubismBreath.create();

    const breathParameters: csmVector<BreathParameterData> = new csmVector();
    
    if (this._idParamAngleX) {
      breathParameters.pushBack(
        new BreathParameterData(this._idParamAngleX, 0.0, 15.0, 6.5345, 0.5)
      );
    }
    if (this._idParamAngleY) {
      breathParameters.pushBack(
        new BreathParameterData(this._idParamAngleY, 0.0, 8.0, 3.5345, 0.5)
      );
    }
    if (this._idParamAngleZ) {
      breathParameters.pushBack(
        new BreathParameterData(this._idParamAngleZ, 0.0, 10.0, 5.5345, 0.5)
      );
    }
    if (this._idParamBodyAngleX) {
      breathParameters.pushBack(
        new BreathParameterData(this._idParamBodyAngleX, 0.0, 4.0, 15.5345, 0.5)
      );
    }

    this._breath.setParameters(breathParameters);
    console.log('[SimpleLive2DModel] Breath setup');
  }

  /**
   * まばたき用パラメータIDをセットアップ
   */
  private setupEyeBlinkIds(): void {
    if (!this._modelSetting) return;

    const eyeBlinkIdCount = this._modelSetting.getEyeBlinkParameterCount();

    for (let i = 0; i < eyeBlinkIdCount; i++) {
      this._eyeBlinkIds.pushBack(this._modelSetting.getEyeBlinkParameterId(i));
    }

    console.log('[SimpleLive2DModel] EyeBlinkIds setup:', eyeBlinkIdCount);
  }

  /**
   * リップシンク用パラメータIDをセットアップ
   */
  private setupLipSyncIds(): void {
    if (!this._modelSetting) return;

    const lipSyncIdCount = this._modelSetting.getLipSyncParameterCount();

    for (let i = 0; i < lipSyncIdCount; i++) {
      this._lipSyncIds.pushBack(this._modelSetting.getLipSyncParameterId(i));
    }

    console.log('[SimpleLive2DModel] LipSyncIds setup:', lipSyncIdCount);
  }

  /**
   * モーションを読み込む
   */
  private async loadMotions(): Promise<void> {
    if (!this._modelSetting) return;

    // すべてのモーショングループを読み込む
    const motionGroups = ['Idle', 'TapBody', 'TapHead', 'Shake', 'Flick'];
    
    for (const group of motionGroups) {
      const motionCount = this._modelSetting.getMotionCount(group);
      if (motionCount === 0) continue;
      
      console.log(`[SimpleLive2DModel] Loading ${motionCount} ${group} motions`);

      for (let i = 0; i < motionCount; i++) {
        const motionFileName = this._modelSetting.getMotionFileName(group, i);
        const name = `${group}_${i}`;

        try {
          const response = await fetch(`${this._modelHomeDir}${motionFileName}`);
          const arrayBuffer = await response.arrayBuffer();

          const motion = CubismMotion.create(arrayBuffer, arrayBuffer.byteLength);
          if (motion) {
            const fadeInTime = this._modelSetting.getMotionFadeInTimeValue(group, i);
            const fadeOutTime = this._modelSetting.getMotionFadeOutTimeValue(group, i);

            if (fadeInTime !== -1.0) {
              motion.setFadeInTime(fadeInTime);
            }
            if (fadeOutTime !== -1.0) {
              motion.setFadeOutTime(fadeOutTime);
            }

            // まばたき・リップシンクのIDを設定
            motion.setEffectIds(this._eyeBlinkIds, this._lipSyncIds);

            this._motions.setValue(name, motion);
            console.log('[SimpleLive2DModel] Motion loaded:', name);
          }
        } catch (error) {
          console.warn('[SimpleLive2DModel] Failed to load motion:', name, error);
        }
      }
    }
  }

  /**
   * レンダラーを作成して初期化
   */
  public setupRenderer(gl: WebGLRenderingContext, width: number, height: number): void {
    this._gl = gl;

    // レンダラーを作成
    this.createRenderer(width, height);

    console.log('[SimpleLive2DModel] Renderer created');

    // テクスチャをセットアップ
    this.setupTextures();

    // レンダラーを起動
    this.getRenderer().startUp(gl);

    console.log('[SimpleLive2DModel] Renderer initialized');
  }

  /**
   * テクスチャをセットアップ
   */
  private setupTextures(): void {
    if (!this._modelSetting || !this._gl) return;

    const textureCount = this._modelSetting.getTextureCount();

    console.log('[SimpleLive2DModel] Loading', textureCount, 'textures');

    for (let i = 0; i < textureCount; i++) {
      const texturePath = this._modelSetting.getTextureFileName(i);
      const textureUrl = `${this._modelHomeDir}${texturePath}`;

      console.log('[SimpleLive2DModel] Loading texture:', textureUrl);

      // テクスチャを読み込み（非同期）
      this.loadTexture(i, textureUrl);
    }
  }

  /**
   * テクスチャを読み込む
   */
  private loadTexture(index: number, textureUrl: string): void {
    if (!this._gl) return;

    const gl = this._gl;
    const img = new Image();

    img.onload = () => {
      // テクスチャオブジェクトを作成
      const tex = gl.createTexture();
      if (!tex) {
        console.error('[SimpleLive2DModel] Failed to create texture');
        return;
      }

      // テクスチャをバインド
      gl.bindTexture(gl.TEXTURE_2D, tex);

      // ミップマップを生成
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

      // Premult alphaを有効にする
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, 1);

      // テクスチャにピクセルを書き込む
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);

      // ミップマップを生成
      gl.generateMipmap(gl.TEXTURE_2D);

      // レンダラーにテクスチャを設定
      this.getRenderer().bindTexture(index, tex);

      this._textures[index] = tex;

      console.log('[SimpleLive2DModel] Texture loaded:', index, textureUrl);
    };

    img.onerror = () => {
      console.error('[SimpleLive2DModel] Failed to load texture:', textureUrl);
    };

    img.src = textureUrl;
  }

  /**
   * モデルを更新
   */
  public update(deltaTimeSeconds: number): void {
    if (this._state !== LoadStep.CompleteSetup) return;
    if (!this._model) return;

    this._userTimeSeconds += deltaTimeSeconds;

    // パラメータを読み込み
    this._model.loadParameters();

    // モーションによるパラメータ更新
    let motionUpdated = false;

    if (this._motionManager.isFinished()) {
      // 最後のモーション実行から一定時間経過後のみアイドルモーションを再生
      const timeSinceLastMotion = this._userTimeSeconds - this._lastMotionTime;
      if (timeSinceLastMotion >= this._idleDelay) {
        // 待機モーションをランダムで再生（遅延時間経過後）
        this.startRandomMotion('Idle', 3);
      }
    } else {
      motionUpdated = this._motionManager.updateMotion(this._model, deltaTimeSeconds);
    }

    // パラメータを保存
    this._model.saveParameters();

    // まばたき
    if (!motionUpdated && this._eyeBlink) {
      this._eyeBlink.updateParameters(this._model, deltaTimeSeconds);
    }

    // 表情
    if (this._expressionManager) {
      this._expressionManager.updateMotion(this._model, deltaTimeSeconds);
    }

    // 呼吸
    if (this._breath) {
      this._breath.updateParameters(this._model, deltaTimeSeconds);
    }

    // 物理演算
    if (this._physics) {
      this._physics.evaluate(this._model, deltaTimeSeconds);
    }

    // ポーズ
    if (this._pose) {
      this._pose.updateParameters(this._model, deltaTimeSeconds);
    }

    // モデルを更新
    this._model.update();
  }

  /**
   * 指定されたモーションファイルを直接再生
   * @param motionFileName モーションファイル名（例: "haru_g_m01"）
   * @param priority 優先度
   */
  public async playMotionByFile(motionFileName: string, priority: number): Promise<void> {
    if (!this._modelSetting) return;

    // .motion3.json を自動的に追加
    const fullFileName = motionFileName.endsWith('.motion3.json') 
      ? motionFileName 
      : `${motionFileName}.motion3.json`;

    // モーションファイルのURLを構築
    const motionUrl = `${this._modelHomeDir}motions/${fullFileName}`;

    console.log(`[SimpleLive2DModel] Loading motion file: ${motionUrl}`);

    try {
      const response = await fetch(motionUrl);
      const arrayBuffer = await response.arrayBuffer();

      const motion = CubismMotion.create(arrayBuffer, arrayBuffer.byteLength);
      if (!motion) {
        console.error(`[SimpleLive2DModel] Failed to create motion from: ${motionUrl}`);
        return;
      }

      // フェードイン/アウト時間を設定（デフォルト値）
      motion.setFadeInTime(0.5);
      motion.setFadeOutTime(0.5);

      // まばたき・リップシンクのIDを設定（重要！）
      motion.setEffectIds(this._eyeBlinkIds, this._lipSyncIds);

      // モーションを再生
      const motionHandle = this._motionManager.startMotionPriority(
        motion,
        false,
        priority
      );

      if (motionHandle) {
        console.log(`[SimpleLive2DModel] Started motion: ${motionFileName} (priority: ${priority})`);
        // 優先度の高いモーション（priority >= 5）の場合、最後の実行時刻を更新
        if (priority >= 5) {
          this._lastMotionTime = this._userTimeSeconds;
        }
      } else {
        console.warn(`[SimpleLive2DModel] Failed to start motion: ${motionFileName}`);
      }
    } catch (error) {
      console.error(`[SimpleLive2DModel] Error loading motion file ${motionUrl}:`, error);
    }
  }

  /**
   * ランダムなモーションを開始
   */
  public startRandomMotion(group: string, priority: number): void {
    if (!this._modelSetting) return;

    const motionCount = this._modelSetting.getMotionCount(group);
    if (motionCount === 0) return;

    const no = Math.floor(Math.random() * motionCount);
    const name = `${group}_${no}`;

    const motion = this._motions.getValue(name);
    if (motion) {
      this._motionManager.startMotionPriority(motion, false, priority);
      console.log('[SimpleLive2DModel] Started motion:', name);
      
      // Idle以外のモーションの場合、最後の実行時刻を更新
      if (group !== 'Idle' && priority >= 5) {
        this._lastMotionTime = this._userTimeSeconds;
      }
    }
  }

  /**
   * 表情を設定
   */
  public setExpression(expressionName: string): void {
    const expression = this._expressions.getValue(expressionName);
    if (expression) {
      this._expressionManager.startMotionPriority(expression, false, 1);
      console.log('[SimpleLive2DModel] Set expression:', expressionName);
    } else {
      console.warn('[SimpleLive2DModel] Expression not found:', expressionName);
    }
  }

  /**
   * モデルを描画
   */
  public draw(projection: CubismMatrix44): void {
    if (this._state !== LoadStep.CompleteSetup) return;
    if (!this._model || !this._gl) return;

    const renderer = this.getRenderer();
    if (!renderer) return;

    // モデル行列を設定
    const modelMatrix = this.getModelMatrix();

    // projection行列とモデル行列を合成
    const mvp = new CubismMatrix44();
    mvp.multiplyByMatrix(projection);
    mvp.multiplyByMatrix(modelMatrix);

    // レンダラーに行列を設定
    renderer.setMvpMatrix(mvp);

    // 描画
    renderer.drawModel();
  }

  /**
   * テクスチャを解放
   */
  public releaseTextures(): void {
    if (!this._gl) return;

    for (const texture of this._textures) {
      this._gl.deleteTexture(texture);
    }

    this._textures = [];
  }

  /**
   * モデルを解放
   */
  public release(): void {
    this.releaseTextures();

    // レンダラーを解放
    this.deleteRenderer();

    // エフェクトをクリア（基底クラスのプロパティなので直接操作しない）

    // モデルを解放
    if (this._model) {
      this._model.release();
    }

    // 物理演算を解放
    if (this._physics) {
      this._physics.release();
    }

    // 表情とモーションを解放
    this._expressions.clear();
    this._motions.clear();
  }
}
