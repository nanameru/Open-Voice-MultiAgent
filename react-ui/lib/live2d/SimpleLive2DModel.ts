/**
 * SimpleLive2DModel
 * CubismUserModelを継承した最小限のLive2Dモデルクラス
 */

import { CubismUserModel } from '@framework/model/cubismusermodel';
import { CubismModelSettingJson } from '@framework/cubismmodelsettingjson';
import { ICubismModelSetting } from '@framework/icubismmodelsetting';
import { CubismPhysics } from '@framework/physics/cubismphysics';
import { CubismMatrix44 } from '@framework/math/cubismmatrix44';

enum LoadStep {
  LoadAssets,
  LoadModel,
  WaitLoadModel,
  LoadPhysics,
  WaitLoadPhysics,
  LoadTexture,
  WaitLoadTexture,
  CompleteSetup,
}

export class SimpleLive2DModel extends CubismUserModel {
  private _modelHomeDir: string = '';
  private _modelSetting: ICubismModelSetting | null = null;
  private _state: LoadStep = LoadStep.LoadAssets;
  private _userTimeSeconds: number = 0.0;
  private _textures: WebGLTexture[] = [];
  private _gl: WebGLRenderingContext | null = null;

  /**
   * model3.jsonが置かれたディレクトリとファイルパスからモデルを生成する
   * @param dir モデルのディレクトリ
   * @param fileName model3.jsonのファイル名
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

    // 物理演算を読み込み
    await this.loadPhysics();

    this._state = LoadStep.CompleteSetup;
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

    // 物理演算を適用
    if (this._physics) {
      this._physics.evaluate(this._model, deltaTimeSeconds);
    }

    // パラメータを保存
    this._model.saveParameters();

    // モデルを更新
    this._model.update();
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

    // モデルを解放
    if (this._model) {
      this._model.release();
    }

    // 物理演算を解放
    if (this._physics) {
      this._physics.release();
    }
  }
}

