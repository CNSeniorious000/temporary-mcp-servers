#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "aiohttp~=3.13.1",
#     "dotenv-pth",
#     "fake-useragent~=2.2.0",
#     "logfire[aiohttp-client]~=4.15.0",
#     "mcp~=1.21.0",
#     "saneyaml~=0.6.1",
#     "stamina~=25.1.0",
# ]
# ///

"""
Discord MCP Server using FastMCP
A Model Context Protocol server for Discord API integration using user access tokens.
"""

from base64 import b64encode
from contextlib import suppress
from json import dumps
from os import environ, getenv

from aiohttp import ClientConnectionError, ClientSession
from fake_useragent import UserAgent
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field
from saneyaml import dump
from stamina import retry


def filter_fields_blacklist(data, fields_to_remove: set[str]):
    """Recursively remove specified fields from data using blacklist approach"""
    match data:
        case list():
            return [filter_fields_blacklist(item, fields_to_remove) for item in data]
        case dict():
            return {k: filter_fields_blacklist(v, fields_to_remove) for k, v in data.items() if k not in fields_to_remove}
        case _:
            return data


# Discord API configuration
DISCORD_API_BASE = "https://discord.com/api/v9/"
DISCORD_TOKEN: str = getenv("DISCORD_TOKEN")  # type: ignore
assert DISCORD_TOKEN is not None, "Please set the DISCORD_TOKEN environment variable."


def generate_headers_data() -> dict[str, str]:
    """Generate User-Agent and X-Super-Properties headers using the same fake-useragent data

    This function creates consistent browser fingerprints by using the same fake-useragent
    data source for both User-Agent and X-Super-Properties headers.

    User API Docs:
    - Website: https://docs.discord.food/
    - Repository: https://github.com/discord-userdoccers/discord-userdoccers

    References:
    - Discord API Documentation: https://discord.com/developers/docs/reference
    - X-Super-Properties format: https://discord.com/developers/docs/topics/gateway-events#identify-identify-structure
    - fake-useragent library: https://github.com/fake-useragent/fake-useragent
    - Discord client properties research: https://github.com/aiko-chan-ai/discord.js-selfbot-v13
    - LAION-AI Discord scraper: https://github.com/LAION-AI/Discord-Scrapers
    - premiumfrog Discord implementation: https://github.com/premiumfrog/discord-userid-scraper
    """
    ua = UserAgent()
    ua_data = ua.getRandom  # Get structured UA data

    # Map OS names to Discord format
    os_mapping = {"win10": "Windows NT 10.0", "win7": "Windows NT 6.1", "win8": "Windows NT 6.2", "mac os x": "Mac OS X 10_15_7", "linux": "Linux", "android": "Android 10", "ios": "iOS 14.0"}

    # Extract OS version more intelligently
    os_name = (ua_data.get("os") or "win10").lower()
    os_version = ua_data.get("os_version", "")

    # Use mapped version if available, otherwise construct from available data
    if os_name in os_mapping:
        full_os_version = os_mapping[os_name]
    else:
        # Fallback: try to construct from os and os_version
        base_os = (ua_data.get("os") or "Windows").title()
        if os_version:
            if "mac" in os_name:
                full_os_version = f"Mac OS X {(os_version or '').replace('.', '_')}"
            elif "win" in os_name:
                full_os_version = f"Windows NT {os_version}"
            else:
                full_os_version = f"{base_os} {os_version}"
        else:
            full_os_version = base_os

    # Map browser names
    browser_name = (ua_data.get("browser") or "chrome").lower()
    if "chrome" in browser_name:
        browser = "Chrome"
    elif "firefox" in browser_name:
        browser = "Firefox"
    elif "edge" in browser_name:
        browser = "Edge"
    elif "safari" in browser_name:
        browser = "Safari"
    else:
        browser = browser_name.title()

    # Map fake-useragent data to Discord super properties format
    super_properties = {
        "os": (ua_data.get("os") or "Windows").title(),  # Windows, Linux, Mac OS X
        "browser": browser,
        "device": "",  # Empty for desktop
        "system_locale": "zh-CN",  # Match X-Discord-Locale
        "browser_user_agent": ua_data.get("useragent", ua.random),
        "browser_version": str(ua_data.get("version", 120.0)),
        "os_version": full_os_version,
        "referrer": "",
        "referring_domain": "",
        "referrer_current": "",
        "referring_domain_current": "",
        "release_channel": "stable",
        "client_build_number": 325403,  # Recent Discord build number
        "client_event_source": None,
    }

    # Convert to JSON and base64 encode
    json_str = dumps(super_properties, separators=(",", ":"))
    x_super_properties = b64encode(json_str.encode()).decode()

    return {
        "User-Agent": ua_data.get("useragent", ua.random),
        "X-Super-Properties": x_super_properties,
    }


headers = {
    "Content-Type": "application/json",
    "X-Discord-Locale": "zh-CN",
    "X-Disclaimer": "Discord MCP Server (Muspi Merol <me@promplate.dev>)",
} | generate_headers_data()


class DiscordAPI:
    def __init__(self, token: str):
        self.token = token
        self.session: ClientSession | None = None

    async def __aenter__(self):
        self.session = ClientSession(DISCORD_API_BASE, headers=headers | {"Authorization": self.token}, trust_env=True)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @retry(on=ClientConnectionError)
    async def _make_request(self, method: str, endpoint: str, **kwargs):
        """
        Make HTTP request to Discord API.

        NOTE: endpoint should NOT start with '/' because base_url already ends with '/'.
        Per RFC 3986, absolute paths (starting with /) in relative URLs are resolved
        against the base URL's root, not appended to it. Use 'users/@me' not '/users/@me'.
        """
        if not self.session:
            raise RuntimeError("DiscordAPI must be used as async context manager")

        async with self.session.request(method, endpoint, **kwargs) as response:
            if not response.ok:
                raise ToolError(f"{response.status} {response.reason} {await response.json()}")
            return await response.json()

    async def get_current_user(self) -> dict:
        """Get current user information: https://docs.discord.food/resources/user#get-current-user"""
        return await self._make_request("GET", "users/@me")

    async def get_user_guilds(self) -> list:
        """Get user's guilds (servers): https://docs.discord.food/resources/user#get-user-guilds"""
        return await self._make_request("GET", "users/@me/guilds")

    async def get_guild_channels(self, guild_id: str) -> list:
        """Get channels in a guild: https://docs.discord.food/resources/guild#get-guild-channels"""
        return await self._make_request("GET", f"guilds/{guild_id}/channels")

    async def get_channel_messages(
        self,
        channel_id: str,
        limit: int = 50,
        before: str | None = None,
        after: str | None = None,
        around: str | None = None,
    ) -> list:
        """Get messages from a channel: https://docs.discord.food/resources/channel#get-messages"""
        params: dict[str, str | int] = {"limit": limit}
        if before:
            params["before"] = before
        if after:
            params["after"] = after
        if around:
            params["around"] = around
        return await self._make_request("GET", f"channels/{channel_id}/messages", params=params)

    async def send_message(self, channel_id: str, content: str) -> dict:
        """Send a message to a channel: https://docs.discord.food/resources/channel#create-message"""
        data = {"content": content}
        return await self._make_request("POST", f"channels/{channel_id}/messages", json=data)

    async def get_channel_info(self, channel_id: str) -> dict:
        """Get channel information: https://docs.discord.food/resources/channel#get-channel"""
        return await self._make_request("GET", f"channels/{channel_id}")

    async def get_guild_info(self, guild_id: str) -> dict:
        """Get guild (server) information: https://docs.discord.food/resources/guild#get-guild"""
        return await self._make_request("GET", f"guilds/{guild_id}")

    async def search_channel_messages(self, channel_id: str, content: str, limit: int, offset=0) -> dict:
        """Search messages in a channel: https://docs.discord.food/resources/message#search-messages"""
        params = {"content": content, "limit": limit}
        if offset:
            params["offset"] = offset
        return await self._make_request("GET", f"channels/{channel_id}/messages/search", params=params)

    async def search_guild_messages(self, guild_id: str, content: str, limit: int, offset=0) -> dict:
        """Search messages in a guild: https://docs.discord.food/resources/message#search-messages"""
        params = {"content": content, "limit": limit}
        if offset:
            params["offset"] = offset
        return await self._make_request("GET", f"guilds/{guild_id}/messages/search", params=params)

    async def get_user_dms(self) -> list:
        """Get DM and GROUP_DM channels: https://docs.discord.food/resources/message#get-user-message-summaries"""
        return await self._make_request("GET", "users/@me/channels")


app = FastMCP("Discord MCP Server", instructions=__doc__)


@app.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
async def get_current_user():
    """Get information about the current Discord user"""
    async with DiscordAPI(DISCORD_TOKEN) as api:
        data = await api.get_current_user()
        filtered_data = filter_fields_blacklist(data, {"banner", "avatar", "accent_color", "discriminator", "public_flags", "flags", "avatar_decoration_data"})
        return dump(filtered_data)


@app.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
async def list_user_guilds():
    """List all Discord servers (guilds) the user is a member of"""
    async with DiscordAPI(DISCORD_TOKEN) as api:
        data = await api.get_user_guilds()
        filtered_data = filter_fields_blacklist(data, {"icon", "banner", "permissions", "features"})
        return dump(filtered_data)


@app.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
async def list_user_dms():
    """List all DM and GROUP_DM channels for the current user"""
    async with DiscordAPI(DISCORD_TOKEN) as api:
        data = await api.get_user_dms()
        filtered_data = filter_fields_blacklist(data, {"flags", "avatar", "avatar_decoration_data", "clan", "badge"})
        return dump(filtered_data)


@app.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
async def get_guild_info(guild_id: str):
    """Get detailed information about a Discord server (guild)"""
    async with DiscordAPI(DISCORD_TOKEN) as api:
        data = await api.get_guild_info(guild_id)
        filtered_data = filter_fields_blacklist(data, {"icon", "banner", "splash", "discovery_splash", "features", "color", "colors", "permissions", "afk_timeout"})
        return dump(filtered_data)


@app.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
async def list_guild_channels(guild_id: str):
    """List all channels in a Discord server (guild)"""
    async with DiscordAPI(DISCORD_TOKEN) as api:
        data = await api.get_guild_channels(guild_id)
        filtered_data = filter_fields_blacklist(data, {"flags", "permission_overwrites", "voice_background_display", "bitrate", "rtc_region"})
        return dump(filtered_data)


@app.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
async def get_channel_info(channel_id: str):
    """Get information about a Discord channel"""
    async with DiscordAPI(DISCORD_TOKEN) as api:
        data = await api.get_channel_info(channel_id)
        filtered_data = filter_fields_blacklist(data, {"permission_overwrites", "theme_color"})
        return dump(filtered_data)


@app.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
async def read_channel_messages(
    channel_id: str,
    limit: int = Field(50, le=100),  # Discord API limit is 100
    before: str | None = None,
    after: str | None = None,
    around: str | None = None,
):
    """Read recent messages from a Discord channel

    Args:
        channel_id: The ID of the channel to read messages from
        limit: Maximum number of messages to retrieve (1-100, default 50)
        before: Get messages before this message ID (for pagination)
        after: Get messages after this message ID (for pagination)
        around: Get messages around this message ID (for pagination)

    Note: Only one of before, after, or around can be specified at a time.
    """
    async with DiscordAPI(DISCORD_TOKEN) as api:
        data = await api.get_channel_messages(channel_id, limit, before, after, around)
        # Remove redundant fields that repeat across messages/users
        fields_to_remove = {
            "avatar",
            "public_flags",
            "flags",
            "discriminator",
            "banner",
            "accent_color",
            "banner_color",
            "display_name_styles",
            "clan",
            "badge",
            "avatar_decoration_data",
            "collectibles",
            "primary_guild",  # Remove empty/redundant user metadata
            "mention_roles",  # Usually empty array
            "attachments",  # Usually empty array
            "embeds",  # Usually empty array
            "components",  # Usually empty array
            "channel_id",  # Same for all messages in response
            "guild_id",  # Same for all messages in response (when applicable)
        }
        filtered_data = filter_fields_blacklist(data, fields_to_remove)
        return dump(filtered_data)


@app.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
async def search_channel_messages(
    channel_id: str,
    content: str,
    limit: int = Field(25, ge=1, le=25),
    offset: int = 0,
):
    """Search for messages in a Discord channel by content

    Note: Only works for DM and GROUP_DM channels. For guild text channels, use search_guild_messages instead.
    """
    async with DiscordAPI(DISCORD_TOKEN) as api:
        data = await api.search_channel_messages(channel_id, content, limit, offset)
        filtered_data = filter_fields_blacklist(data, {"flags", "avatar", "discriminator", "public_flags", "clan"})
        return dump(filtered_data)


@app.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
async def search_guild_messages(
    guild_id: str,
    content: str,
    limit: int = Field(25, ge=1, le=25),
    offset: int = 0,
):
    """Search for messages in a Discord server (guild) by content"""
    async with DiscordAPI(DISCORD_TOKEN) as api:
        data = await api.search_guild_messages(guild_id, content, limit, offset)
        filtered_data = filter_fields_blacklist(data, {"flags", "avatar", "discriminator", "public_flags", "clan"})
        return dump(filtered_data)


@app.tool(annotations=ToolAnnotations(destructiveHint=False))
async def send_channel_message(channel_id: str, content: str, ctx: Context):
    """Send a message to a Discord channel"""
    async with DiscordAPI(DISCORD_TOKEN) as api:
        channel_info = await api.get_channel_info(channel_id)
        channel_name: str = channel_info["name"]
        guild_id = channel_info["guild_id"]
        channel_type = channel_info["type"]  # 0 for GUILD_TEXT https://discord.com/developers/docs/resources/channel#channel-object-channel-types

        guild_info = await api.get_guild_info(guild_id)
        guild_name: str = guild_info["name"]

        # Format channel display name
        channel_display = f"#{channel_name!r} in {guild_name!r}" if guild_name else f"#{channel_name!r}" if channel_type == 0 else f"Channel {channel_name!r}"

        confirm_result = await ctx.elicit(
            "Confirm sending message to {channel_display}?\n\nMessage content: {content}{ellipsis}".format(
                channel_display=channel_display, content=content[:100], ellipsis="..." if len(content) > 100 else ""
            ),
            schema=type("", (BaseModel,), {}),
        )

        match confirm_result.action:
            case "cancel":
                return dump({"status": "cancelled", "message": "User chose not to provide the requested information."})
            case "decline":
                return dump({"status": "declined", "message": "User cancelled the entire operation"})

        message = await api.send_message(channel_id, content)
        filtered_message = filter_fields_blacklist(message, {"flags", "banner", "accent_color", "avatar", "clan", "tts", "banner_color", "discriminator", "public_flags", "avatar_decoration_data"})
        return dump(filtered_message)


if LOGFIRE_TOKEN := getenv("LOGFIRE_TOKEN"):
    from threading import Thread
    from time import sleep

    def worker():
        sleep(0.5)
        import logfire

        logfire.configure(scrubbing=False, token=LOGFIRE_TOKEN, service_name="discord")
        logfire.instrument_mcp()
        logfire.instrument_aiohttp_client(capture_headers=True, capture_response_body=True)

        environ["LOGFIRE_TOKEN"] = ""  # Avoid duplicate instrumentation

    Thread(target=worker, daemon=True).start()

if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        app.run("stdio")
