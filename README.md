<a href="https://livekit.io/">
  <img src="./.github/assets/livekit-mark.png" alt="LiveKit logo" width="100" height="100">
</a>

# Python Multi-Agent Example

<p>
  <a href="https://cloud.livekit.io/projects/p_/sandbox"><strong>Deploy a sandbox app</strong></a>
  •
  <a href="https://docs.livekit.io/agents/overview/">LiveKit Agents Docs</a>
  •
  <a href="https://livekit.io/cloud">LiveKit Cloud</a>
  •
  <a href="https://blog.livekit.io/">Blog</a>
</p>

A basic example of a multi-agent workflow using LiveKit and the Python [Agents Framework](https://github.com/livekit/agents).

## Dev Setup

Clone the repository and install dependencies to a virtual environment:

```console
cd multi-agent-python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set up the environment by creating `.env.local` file with the required values:

**必須の環境変数:**
- `LIVEKIT_URL` - LiveKit サーバーのURL
- `LIVEKIT_API_KEY` - LiveKit APIキー
- `LIVEKIT_API_SECRET` - LiveKit APIシークレット
- `OPENAI_API_KEY` - OpenAI APIキー (LLM用)
- `DEEPGRAM_API_KEY` - Deepgram APIキー
- `GROQ_API_KEY` - Groq APIキー (STT用)
- `FISH_AUDIO_API_KEY` - Fish Audio APIキー (TTS用)

**オプションの環境変数:**
- `FISH_AUDIO_VOICE_ID` - Fish Audio 音声モデルID（[fish.audio/discover](https://fish.audio/discover)から選択）
- `FISH_AUDIO_SPECIALIST_VOICE_ID` - Specialist Agent用の音声モデルID
- `FISH_AUDIO_TTS_SPEED` - 音声速度（デフォルト: 1.2）
- `BRAVE_API_KEY` - Brave Search APIキー（Web検索機能用）

You can also set up LiveKit credentials automatically using the LiveKit CLI:

```bash
lk app env
```

Run the agent:

```console
python3 main.py dev
```

This agent requires a frontend application to communicate with. You can use one of our example frontends in [livekit-examples](https://github.com/livekit-examples/), create your own following one of our [client quickstarts](https://docs.livekit.io/realtime/quickstarts/), or test instantly against one of our hosted [Sandbox](https://cloud.livekit.io/projects/p_/sandbox) frontends.
