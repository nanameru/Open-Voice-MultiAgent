"""Primary agent orchestrating story ideation."""

import logging
from livekit.agents import Agent, RunContext
from livekit.agents.llm import function_tool

from .constants import COMMON_INSTRUCTIONS
from .models import CharacterData, StoryData
from .specialist_editor_agent import SpecialistEditorAgent

logger = logging.getLogger("multi-agent")


class LeadEditorAgent(Agent):
    """Lead editor that triages user requests and delegates to specialists."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                f"{COMMON_INSTRUCTIONS} You are the lead editor at this business, "
                "and are yourself a generalist -- but empoly several specialist editors, "
                "specializing in childrens' books and fiction, respectively. You trust your "
                "editors to do their jobs, and will hand off the conversation to them when you feel "
                "you have an idea of the right one."
                "Your goal is to gather a few pieces of information from the user about their next"
                "idea for a short story, and then hand off to the right agent."
                "Start the conversation with a short introduction, then get straight to the "
                "details. You may hand off to either editor as soon as you know which one is the right fit."
            ),
        )

    async def on_enter(self):
        """Generate an initial reply when entering the session."""

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
    async def detected_childrens_book(
        self,
        context: RunContext[StoryData],
    ):
        """Hand off to the children's book specialist."""

        childrens_editor = SpecialistEditorAgent("children's books", chat_ctx=context.session._chat_ctx)
        logger.info(
            "switching to the children's book editor with the provided user data: %s", context.userdata
        )
        return childrens_editor, "Let's switch to the children's book editor."

    @function_tool
    async def detected_novel(
        self,
        context: RunContext[StoryData],
    ):
        """Hand off to the novel specialist."""

        novel_editor = SpecialistEditorAgent("novels", chat_ctx=context.session._chat_ctx)
        logger.info(
            "switching to the children's book editor with the provided user data: %s", context.userdata
        )
        return novel_editor, "Let's switch to the children's book editor."
