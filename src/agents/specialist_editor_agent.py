"""Specialist editor agent implementations."""

import logging
from typing import Optional

from livekit import api
from livekit.agents import Agent, ChatContext, RunContext
from livekit.agents.job import get_job_context
from livekit.agents.llm import function_tool
from livekit.plugins import openai

from .constants import COMMON_INSTRUCTIONS
from .models import CharacterData, StoryData

logger = logging.getLogger("multi-agent")


class SpecialistEditorAgent(Agent):
    """Domain specialist editor that refines story ideas."""

    def __init__(self, specialty: str, chat_ctx: Optional[ChatContext] = None) -> None:
        super().__init__(
            instructions=(
                f"{COMMON_INSTRUCTIONS}. You specialize in {specialty}, and have "
                "worked with some of the greats, and have even written a few books yourself."
            ),
            tts=openai.TTS(voice="echo"),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self):
        """Generate a reply when the specialist joins the session."""

        self.session.generate_reply()

    @function_tool
    async def character_introduction(
        self,
        context: RunContext[StoryData],
        name: str,
        background: str,
    ):
        """Called when the user has provided a character."""

        character = CharacterData(name=name, background=background)
        context.userdata.characters.append(character)

        logger.info("added character to the story: %s", name)

    @function_tool
    async def location_introduction(
        self,
        context: RunContext[StoryData],
        location: str,
    ):
        """Called when the user has provided a location."""

        context.userdata.locations.append(location)

        logger.info("added location to the story: %s", location)

    @function_tool
    async def theme_introduction(
        self,
        context: RunContext[StoryData],
        theme: str,
    ):
        """Called when the user has provided a theme."""

        context.userdata.theme = theme

        logger.info("set theme to the story: %s", theme)

    @function_tool
    async def story_finished(self, context: RunContext[StoryData]):
        """Wrap up the session once the story outline is complete."""

        self.session.interrupt()

        await self.session.generate_reply(
            instructions="give brief but honest feedback on the story idea", allow_interruptions=False
        )

        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))
