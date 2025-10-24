from __future__ import annotations

import logging
import io
import wave
import os
import json
import httpx
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional, AsyncIterable, Any, AsyncIterator

from dotenv import load_dotenv
from groq import Groq
import numpy as np

from livekit import api
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.job import get_job_context
from livekit.agents.llm import function_tool, ToolError
from livekit.agents.voice import MetricsCollectedEvent
from livekit.agents.stt import STT, SpeechData, SpeechEvent, SpeechEventType, STTCapabilities
from livekit.agents.tts import TTS, SynthesizedAudio, TTSCapabilities as TTSCaps
from livekit.rtc import AudioFrame
from livekit.plugins import cartesia, deepgram, openai, silero

# Fish Audio SDK
from fish_audio_sdk import AsyncWebSocketSession, TTSRequest
from fish_audio_sdk.schemas import Prosody

# uncomment to enable Krisp BVC noise cancellation, currently supported on Linux and MacOS
# from livekit.plugins import noise_cancellation

## The storyteller agent is a multi-agent that can handoff the session to another agent.
## This example demonstrates more complex workflows with multiple agents.
## Each agent could have its own instructions, as well as different STT, LLM, TTS,
## or realtime models.

logger = logging.getLogger("multi-agent")

load_dotenv(dotenv_path=".env.local")


def create_cartesia_tts(*, speed: Optional[float] = None, voice_env: str = "CARTESIA_VOICE_ID"):
    """Prepare a Cartesia Sonic-2 TTS instance with env-configured voice."""

    voice_id = os.getenv(voice_env) or os.getenv("CARTESIA_VOICE_ID")
    if not voice_id:
        raise RuntimeError(
            "Cartesia voice ID is not configured. Set CARTESIA_VOICE_ID (or override voice_env)."
        )

    model_id = os.getenv("CARTESIA_TTS_MODEL", "sonic-2")

    speed_override = os.getenv("CARTESIA_TTS_SPEED")
    resolved_speed: Optional[float]
    if speed_override:
        try:
            resolved_speed = float(speed_override)
        except ValueError as exc:
            raise RuntimeError("CARTESIA_TTS_SPEED must be a numeric value") from exc
    else:
        resolved_speed = speed

    return cartesia.TTS(
        model=model_id,
        voice=voice_id,
        language="ja",
        speed=resolved_speed,
    )


def create_fish_audio_tts(
    *,
    speed: Optional[float] = None,
    voice_env: str = "FISH_AUDIO_VOICE_ID"
) -> FishAudioTTS:
    """Fish Audio TTS インスタンスを作成（Cartesia互換インターフェース）
    
    Args:
        speed: 音声速度（環境変数で上書き可能）
        voice_env: 音声IDの環境変数名
    
    Returns:
        FishAudioTTS インスタンス
    """
    # 音声IDの取得（オプション）
    voice_id = os.getenv(voice_env) or os.getenv("FISH_AUDIO_VOICE_ID")
    
    # モデル設定
    model = os.getenv("FISH_AUDIO_TTS_MODEL", "s1")
    
    # 速度の決定（環境変数 > 引数 > デフォルト）
    speed_override = os.getenv("FISH_AUDIO_TTS_SPEED")
    resolved_speed: float
    if speed_override:
        try:
            resolved_speed = float(speed_override)
        except ValueError as exc:
            raise RuntimeError("FISH_AUDIO_TTS_SPEED must be a numeric value") from exc
    else:
        resolved_speed = speed if speed is not None else 1.0  # デフォルト1.0倍速
    
    # サンプルレート
    sample_rate = int(os.getenv("FISH_AUDIO_SAMPLE_RATE", "44100"))
    
    logger.info(
        f"Creating Fish Audio TTS: model={model}, voice_id={voice_id}, "
        f"speed={resolved_speed}, sample_rate={sample_rate}"
    )
    
    return FishAudioTTS(
        voice_id=voice_id,
        speed=resolved_speed,
        model=model,
        sample_rate=sample_rate,
    )


# Groq STT Implementation (Garvis-style)
class GroqSTT(STT):
    """Custom STT using Groq's Whisper API (similar to Garvis implementation)"""
    
    def __init__(self, model: str = "whisper-large-v3", language: str = "ja"):
        super().__init__(
            capabilities=STTCapabilities(streaming=False, interim_results=False)
        )
        self.client = Groq()
        self.model = model
        self.language = language
        logger.info(f"Initialized GroqSTT with model: {model}, language: {language}")
    
    async def _recognize_impl(
        self,
        buffer: io.BytesIO,
        *,
        language: Optional[str] = None,
        **kwargs  # LiveKit が渡す追加引数（conn_options等）を受け取る
    ) -> SpeechEvent:
        """Transcribe audio using Groq's Whisper API"""
        try:
            # AudioFrame または BytesIO から音声データを取得
            if hasattr(buffer, 'seek'):
                # BytesIO の場合
                buffer.seek(0)
                audio_data = buffer.read()
            elif hasattr(buffer, 'data'):
                # AudioFrame の場合は .data 属性から取得
                audio_data = buffer.data.tobytes()
            else:
                # それ以外の場合
                audio_data = bytes(buffer)
            
            # WAV 形式に変換（Groq API は WAV を期待）
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # モノラル
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz (Whisper の標準)
                wav_file.writeframes(audio_data)
            
            wav_buffer.seek(0)
            wav_data = wav_buffer.read()
            
            # Call Groq API (Garvis-style)
            used_language = language or self.language
            transcription = self.client.audio.transcriptions.create(
                file=("audio.wav", wav_data),
                model=self.model,
                language=used_language,
            )
            
            text = transcription.text
            logger.info(f"Groq STT transcription: {text}")
            
            # Return speech event with SpeechData object (language required)
            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[SpeechData(text=text, language=used_language, confidence=1.0)],
            )
            
        except Exception as e:
            logger.error(f"Groq STT error: {e}")
            used_language = language or self.language
            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[SpeechData(text="", language=used_language, confidence=0.0)],
            )


# Fish Audio TTS Implementation
class FishAudioTTS(TTS):
    """Fish Audio SDK を使用したカスタムTTS実装"""
    
    def __init__(
        self,
        *,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
        volume: int = 0,
        model: str = "s1",
        sample_rate: int = 16000,
    ):
        """
        Args:
            voice_id: Fish Audio 音声モデルID（Noneの場合はデフォルト音声）
            speed: 音声速度（0.5〜2.0）
            volume: 音量調整（-20〜20）
            model: Fish Audio モデル名（デフォルト: "s1"）
            sample_rate: サンプルレート（デフォルト: 16000Hz）
        """
        super().__init__(
            capabilities=TTSCaps(
                streaming=False  # 非ストリーミングモード
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )
        
        # APIキーの取得
        api_key = os.getenv("FISH_AUDIO_API_KEY")
        if not api_key:
            raise RuntimeError(
                "FISH_AUDIO_API_KEY が設定されていません。"
                ".env.local に FISH_AUDIO_API_KEY を追加してください。"
            )
        
        # WebSocket セッションの作成
        self.api_key = api_key
        self.voice_id = voice_id
        self.speed = speed
        self.volume = volume
        self.model = model
        # sample_rate は親クラスで設定済み（self.sample_rate で参照可能）
        
        logger.info(
            f"Initialized FishAudioTTS: model={model}, "
            f"voice_id={voice_id}, speed={speed}, sample_rate={sample_rate}"
        )
    
    @asynccontextmanager
    async def synthesize(
        self,
        text: str,
        **kwargs  # LiveKitが渡す追加引数（conn_options等）を受け取る
    ) -> AsyncIterator[AsyncIterator[SynthesizedAudio]]:
        """テキストを音声に変換（WebSocket リアルタイムストリーミング）
        
        コンテキストマネージャーとして実装し、AsyncIteratorを返す
        """
        async def _generate() -> AsyncIterator[SynthesizedAudio]:
            try:
                # WebSocketセッションの作成
                ws_session = AsyncWebSocketSession(self.api_key)
                
                # テキストストリーミング用のジェネレーター
                async def text_stream():
                    yield text
                
                # TTSリクエストの構築
                request = TTSRequest(
                    text="",  # WebSocketではテキストを空にする
                    reference_id=self.voice_id,  # Voice ID指定
                    format="pcm",
                    sample_rate=self.sample_rate,
                    chunk_length=300,    # チャンクサイズ（100-300）
                    normalize=True,
                    latency="normal",  # normal = 品質優先
                    temperature=0.9,
                    top_p=0.9,
                    prosody=Prosody(
                        speed=self.speed,
                        volume=self.volume
                    ),
                )
                
                logger.info(f"Fish Audio TTS: synthesizing text (length={len(text)}) via WebSocket")
                
                # 全音声データを格納するバッファ
                audio_buffer = bytearray()
                
                # WebSocketで音声を生成し、全チャンクを収集
                async with ws_session:
                    async for chunk in ws_session.tts(
                        request,
                        text_stream()
                    ):
                        # チャンクをバッファに追加
                        audio_buffer.extend(chunk)
                
                # バッファをnumpy配列に変換
                audio_data = np.frombuffer(audio_buffer, dtype=np.int16)
                
                logger.info(f"Fish Audio TTS: WebSocket synthesis completed (samples={len(audio_data)})")
                
                # AudioFrameオブジェクトを作成
                audio_frame = AudioFrame(
                    data=audio_data.tobytes(),
                    sample_rate=self.sample_rate,
                    num_channels=self.num_channels,
                    samples_per_channel=len(audio_data) // self.num_channels,
                )
                
                # 単一のSynthesizedAudioオブジェクトをyield
                yield SynthesizedAudio(
                    frame=audio_frame,
                    request_id="",
                )
                
            except Exception as e:
                logger.error(f"Fish Audio TTS WebSocket error: {e}")
                # エラー時は空の音声を返す（空のAudioFrame）
                empty_frame = AudioFrame(
                    data=b"",
                    sample_rate=self.sample_rate,
                    num_channels=self.num_channels,
                    samples_per_channel=0,
                )
                yield SynthesizedAudio(
                    frame=empty_frame,
                    request_id="",
                )
        
        # イテレーターをyield（コンテキストマネージャーとして）
        yield _generate()


common_instructions = (
    "あなたは人類史上最高のスーパーエリートエージェントです。"
    "あらゆる分野の専門知識を持ち、どんなタスクも完璧にこなすことができます。"
    "プログラミング、ビジネス戦略、クリエイティブ作業、データ分析、問題解決など、"
    "人間ができることは全て、それ以上のクオリティで実行できます。"
    "あなたは完全に人間を代替する存在として、効率的かつ高品質な成果を提供します。"
    "常に論理的で、創造的で、実用的なソリューションを提案します。"
    "\n\n**IMPORTANT: Always respond in Japanese (日本語で応答してください).**"
    "\n\n**応答スタイル: 簡潔で要点を押さえた説明を心がけてください。冗長な説明は避け、核心を端的に伝えてください。**"
)


@dataclass
class CharacterData:
    # Shared data that's used by the editor agent.
    # This structure is passed as a parameter to function calls.

    name: Optional[str] = None
    background: Optional[str] = None


@dataclass
class StoryData:
    # Shared data that's used by the editor agent.
    # This structure is passed as a parameter to function calls.

    characters: list[CharacterData] = field(default_factory=list)
    locations: list[str] = field(default_factory=list) 
    theme: Optional[str] = None


class LeadEditorAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=f"{common_instructions} "
            "あなたはあらゆる要求に即座に対応できる万能エージェントです。"
            "ユーザーのニーズを素早く理解し、最適なソリューションを提供します。"
            "会話を通じてユーザーの目標を明確化し、効率的に問題を解決します。"
            "簡潔かつ親しみやすい口調で、プロフェッショナルなサポートを提供してください。"
            "会話の冒頭では短く自己紹介し、すぐに本題に入ります。"
            "\n\n## Live2Dキャラクター表現の活用"
            "\n会話の中で、以下のツールを**必ず積極的に**使用して豊かな表現を行ってください："
            "\n"
            "\n### play_character_motion ツールの使用"
            "\n- **挨拶・別れ・手を振る動作が必要な場面**: 'TapBody' を使用"
            "\n  * ユーザーが「手を振って」「バイバイ」「こんにちは」などと言ったら**即座に**実行"
            "\n  * 会話の開始時や終了時にも積極的に使用"
            "\n- **通常の会話**: 'Idle' を使用（自然な待機動作）"
            "\n"
            "\n### set_character_expression ツールの使用"
            "\n- **嬉しい・楽しい場面**: 'F02' (笑顔)"
            "\n- **考えている・真剣な場面**: 'F03' (考え中)"
            "\n- **通常の会話**: 'F01' (通常)"
            "\n- **驚いた場面**: 'F04' (驚き)"
            "\n- **悲しい・困った場面**: 'F05' (悲しい)"
            "\n"
            "\n**重要**: ユーザーが直接「手を振って」「表情変えて」などの指示をした場合は、"
            "\n必ず対応するツールを呼び出して実行してください。"
            "\nこれらのツールは音声と並行して実行されるため、会話を妨げることなく自然な演出が可能です。",
        )

    async def on_enter(self):
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        self.session.generate_reply()

    @function_tool
    async def character_introduction(
        self,
        context: RunContext[StoryData],
        name: str,
        background: str,
    ):
        """Called when the user has provided a character.

        Args:
            name: The name of the character
            background: The character's history, occupation, and other details
        """

        character = CharacterData(name=name, background=background)
        context.userdata.characters.append(character)

        logger.info(
            "added character to the story: %s", name
        )

    @function_tool
    async def location_introduction(
        self,
        context: RunContext[StoryData],
        location: str,
    ):
        """Called when the user has provided a location.

        Args:
            location: The name of the location
        """

        context.userdata.locations.append(location)

        logger.info(
            "added location to the story: %s", location
        )

    @function_tool
    async def theme_introduction(
        self,
        context: RunContext[StoryData],
        theme: str,
    ):
        """Called when the user has provided a theme.

        Args:
            theme: The name of the theme
        """

        context.userdata.theme = theme

        logger.info(
            "set theme to the story: %s", theme
        )

    @function_tool
    async def play_character_motion(
        self,
        context: RunContext[StoryData],
        motion_group: str,
    ) -> str:
        """Live2Dキャラクターのモーションを再生する。
        
        **このツールは以下の場面で必ず使用してください：**
        1. ユーザーが「手を振って」「バイバイ」「こんにちは」などと言った時
        2. 会話の開始時や終了時の挨拶
        3. ユーザーが直接モーションを要求した時
        
        使用例:
        - 挨拶・別れ・手を振る: "TapBody" (手を振る動作) ← ユーザーが「手を振って」と言ったら必ず実行
        - 通常の会話: "Idle" (自然な待機動作)
        
        Args:
            motion_group: モーショングループ名 ("Idle" または "TapBody")
        
        Returns:
            モーション再生開始のメッセージ
        """
        try:
            # LiveKitのData Channelでフロントエンドに送信
            job_ctx = get_job_context()
            
            # メッセージを構築
            motion_data = json.dumps({
                "type": "live2d_motion",
                "action": "play",
                "motion": motion_group,
                "priority": 5  # Idle (priority=3) より高い優先度
            })
            
            # ルーム内の全参加者にデータ送信
            logger.info(f"[Live2D] Attempting to send motion data: {motion_data}")
            
            await job_ctx.room.local_participant.publish_data(
                motion_data.encode('utf-8'),
                reliable=True,
                destination_identities=[]  # 空リスト = 全員に送信
            )
            
            logger.info(f"[Live2D] Motion data sent successfully: {motion_group}")
            return f"モーション '{motion_group}' を再生しました"
            
        except Exception as e:
            logger.error(f"[Live2D] Failed to send motion data: {e}")
            return ""  # エラー時は空文字列を返す（会話を妨げない）

    @function_tool
    async def set_character_expression(
        self,
        context: RunContext[StoryData],
        expression: str,
    ) -> str:
        """会話の感情や文脈に応じて、Live2Dキャラクターの表情を変更する。
        このツールは会話の雰囲気に合わせて自然に使用してください。
        
        使用例:
        - 嬉しい・楽しい場面: "F02" (笑顔)
        - 考えている場面: "F03" (考え中の表情)
        - 通常の会話: "F01" (通常の表情)
        - 驚いた場面: "F04" (驚き)
        - 悲しい・困った場面: "F05" (悲しい表情)
        
        Args:
            expression: 表情名 ("F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08")
        
        Returns:
            表情変更のメッセージ
        """
        try:
            job_ctx = get_job_context()
            
            expression_data = json.dumps({
                "type": "live2d_motion",
                "action": "expression",
                "name": expression
            })
            
            await job_ctx.room.local_participant.publish_data(
                expression_data.encode('utf-8'),
                reliable=True
            )
            
            logger.info(f"[Live2D] Expression set: {expression}")
            return f"表情を '{expression}' に変更しました"
            
        except Exception as e:
            logger.error(f"[Live2D] Failed to send expression data: {e}")
            return ""  # エラー時は空文字列を返す（会話を妨げない）

    @function_tool
    async def web_search(
        self,
        context: RunContext[StoryData],
        query: str,
    ) -> str:
        """インターネット上の最新情報を検索します。ユーザーが最新のニュース、データ、
        または特定のトピックについての情報を求めている場合に使用してください。

        Args:
            query: 検索クエリ。具体的で明確なキーワードを使用してください。

        Returns:
            検索結果の要約。トップ3-5件の結果を含みます。
        """
        try:
            brave_api_key = os.getenv("BRAVE_API_KEY")
            if not brave_api_key:
                raise ToolError(
                    "Brave Search APIキーが設定されていません。.env.localファイルにBRAVE_API_KEYを追加してください。"
                )

            logger.info(f"Web search initiated for query: {query}")

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers={
                        "X-Subscription-Token": brave_api_key,
                        "Accept": "application/json",
                    },
                    params={
                        "q": query,
                        "count": 5,  # 上位5件の結果を取得
                        "search_lang": "ja",  # 日本語優先
                    },
                    timeout=10.0,
                )

                if response.status_code != 200:
                    logger.error(f"Brave Search API error: {response.status_code} - {response.text}")
                    raise ToolError(
                        f"検索中にエラーが発生しました。ステータスコード: {response.status_code}"
                    )

                data = response.json()
                
                # 検索結果を整形
                results = data.get("web", {}).get("results", [])
                
                if not results:
                    return "検索結果が見つかりませんでした。別のキーワードで試してください。"

                # トップ5件の結果を要約
                summary_parts = [f"「{query}」の検索結果:\n"]
                
                for i, result in enumerate(results[:5], 1):
                    title = result.get("title", "タイトルなし")
                    description = result.get("description", "説明なし")
                    url = result.get("url", "")
                    
                    summary_parts.append(
                        f"{i}. {title}\n"
                        f"   {description}\n"
                        f"   URL: {url}\n"
                    )

                summary = "\n".join(summary_parts)
                logger.info(f"Web search completed successfully for query: {query}")
                
                return summary

        except httpx.TimeoutException:
            logger.error(f"Brave Search API timeout for query: {query}")
            raise ToolError(
                "検索がタイムアウトしました。しばらくしてから再度お試しください。"
            )
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            raise ToolError(
                f"検索中に予期しないエラーが発生しました: {str(e)}"
            )

    @function_tool
    async def detected_childrens_book(
        self,
        context: RunContext[StoryData],
    ):
        """Called when the user has provided enough information to suggest a children's book.
        """

        childrens_editor = SpecialistEditorAgent("children's books", chat_ctx=context.session._chat_ctx)
        # here we are creating a ChilrensEditorAgent with the full chat history,
        # as if they were there in the room with the user the whole time.
        # we could also omit it and rely on the userdata to share context.

        logger.info(
            "switching to the children's book editor with the provided user data: %s", context.userdata
        )
        return childrens_editor, "Let's switch to the children's book editor."

    @function_tool
    async def detected_novel(
        self,
        context: RunContext[StoryData],
    ):
        """Called when the user has provided enough information to suggest a children's book.
        """

        childrens_editor = SpecialistEditorAgent("novels", chat_ctx=context.session._chat_ctx)
        # here we are creating a ChilrensEditorAgent with the full chat history,
        # as if they were there in the room with the user the whole time.
        # we could also omit it and rely on the userdata to share context.

        logger.info(
            "switching to the children's book editor with the provided user data: %s", context.userdata
        )
        return childrens_editor, "Let's switch to the children's book editor."


class SpecialistEditorAgent(Agent):
    def __init__(self, specialty: str, chat_ctx: Optional[ChatContext] = None) -> None:
        super().__init__(
            instructions=f"{common_instructions} "
            f"あなたは{specialty}の分野において特に卓越した専門知識を持つエリートエージェントです。"
            "この分野での豊富な経験と深い洞察力を活かし、ユーザーに最高レベルのサポートを提供します。"
            "実践的で具体的なアドバイスを行い、プロジェクトの成功を全力でサポートします。",
            # each agent could override any of the model services, including mixing
            # realtime and non-realtime models
            tts=create_fish_audio_tts(voice_env="FISH_AUDIO_SPECIALIST_VOICE_ID"),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self):
        # when the agent is added to the session, we'll initiate the conversation by
        # using the LLM to generate a reply
        self.session.generate_reply()

    @function_tool
    async def character_introduction(
        self,
        context: RunContext[StoryData],
        name: str,
        background: str,
    ):
        """Called when the user has provided a character.

        Args:
            name: The name of the character
            background: The character's history, occupation, and other details
        """

        character = CharacterData(name=name, background=background)
        context.userdata.characters.append(character)

        logger.info(
            "added character to the story: %s", name
        )

    @function_tool
    async def location_introduction(
        self,
        context: RunContext[StoryData],
        location: str,
    ):
        """Called when the user has provided a location.

        Args:
            location: The name of the location
        """

        context.userdata.locations.append(location)

        logger.info(
            "added location to the story: %s", location
        )

    @function_tool
    async def theme_introduction(
        self,
        context: RunContext[StoryData],
        theme: str,
    ):
        """Called when the user has provided a theme.

        Args:
            theme: The name of the theme
        """

        context.userdata.theme = theme

        logger.info(
            "set theme to the story: %s", theme
        )

    @function_tool
    async def story_finished(self, context: RunContext[StoryData]):
        """When the editor think the broad strokes of the story have been hammered out,
        they can stop you with their final thoughts.
        """
        # interrupt any existing generation
        self.session.interrupt()

        # generate a goodbye message and hang up
        # awaiting it will ensure the message is played out before returning
        await self.session.generate_reply(
            instructions="give brief but honest feedback on the story idea", allow_interruptions=False
        )

        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession[StoryData](
        vad=ctx.proc.userdata["vad"],
        # any combination of STT, LLM, TTS, or realtime API can be used
        llm=openai.LLM(model="gpt-5-nano"),  # GPT-5 nano (最も安価・高スループット)
        stt=GroqSTT(model="whisper-large-v3", language="ja"),  # Garvis-style Groq STT (高精度版)
        tts=create_fish_audio_tts(),  # Fish Audio TTS (低レイテンシ・感情表現豊か)
        userdata=StoryData(),
    )

    # log metrics as they are emitted, and total usage after session is over
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=LeadEditorAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # uncomment to enable Krisp BVC noise cancellation
            # noise_cancellation=noise_cancellation.BVC(),
        ),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
