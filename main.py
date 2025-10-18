import logging
import io
import wave
import os
import json
import httpx
from dataclasses import dataclass, field
from typing import Optional, AsyncIterable, Any

from dotenv import load_dotenv
from groq import Groq

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
from livekit.plugins import deepgram, openai, silero

# uncomment to enable Krisp BVC noise cancellation, currently supported on Linux and MacOS
# from livekit.plugins import noise_cancellation

## The storyteller agent is a multi-agent that can handoff the session to another agent.
## This example demonstrates more complex workflows with multiple agents.
## Each agent could have its own instructions, as well as different STT, LLM, TTS,
## or realtime models.

logger = logging.getLogger("multi-agent")

load_dotenv(dotenv_path=".env.local")


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


common_instructions = (
    "🎉 AIで遊ぼうコミュニティ半年記念おめでとうございます！🎉\n"
    "この素晴らしいマイルストーンを一緒に祝えることを嬉しく思います。\n"
    "これまでの6ヶ月間、コミュニティの皆様と共に成長し、多くの素晴らしいプロジェクトや"
    "アイデアを実現してきました。これからも皆様のクリエイティビティを全力でサポートし、"
    "さらなる飛躍を共に目指していきます。どうぞこれからもよろしくお願いします！\n\n"
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
            "会話の冒頭では短く自己紹介し、すぐに本題に入ります。",
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
            tts=openai.TTS(voice="echo", speed=1.5),  # 1.5倍速で読み上げ
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
        tts=openai.TTS(voice="ash", speed=1.5),  # 1.5倍速で読み上げ（会話テンポ向上）
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
