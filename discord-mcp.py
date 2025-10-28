#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "aiohttp~=3.13.1",
#     "dotenv-pth",
#     "fake-useragent~=2.2.0",
#     "logfire[aiohttp-client]~=4.14.1",
#     "mcp~=1.19.0",
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
from pydantic import BaseModel, Field
from saneyaml import dump
from stamina import retry

# Discord API configuration
DISCORD_API_BASE = "https://discord.com/api/v9"
DISCORD_TOKEN: str = getenv("DISCORD_TOKEN")  # type: ignore
assert DISCORD_TOKEN is not None, "Please set the DISCORD_TOKEN environment variable."


def generate_headers_data() -> dict[str, str]:
    """Generate User-Agent and X-Super-Properties headers using the same fake-useragent data

    This function creates consistent browser fingerprints by using the same fake-useragent
    data source for both User-Agent and X-Super-Properties headers.

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
        self.session = ClientSession(headers=headers | {"Authorization": self.token})
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @retry(on=ClientConnectionError)
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> dict | list | None:
        if not self.session:
            raise RuntimeError("DiscordAPI must be used as async context manager")

        url = f"{DISCORD_API_BASE}{endpoint}"
        async with self.session.request(method, url, **kwargs) as response:
            if response.status == 204:  # No Content
                return None
            response.raise_for_status()
            return await response.json()

    async def get_current_user(self) -> dict | list | None:
        """Get current user information"""
        return await self._make_request("GET", "/users/@me")

    async def get_user_guilds(self) -> dict | list | None:
        """Get user's guilds (servers)"""
        return await self._make_request("GET", "/users/@me/guilds")

    async def get_guild_channels(self, guild_id: str) -> dict | list | None:
        """Get channels in a guild"""
        return await self._make_request("GET", f"/guilds/{guild_id}/channels")

    async def get_channel_messages(
        self,
        channel_id: str,
        limit: int = Field(50, le=100),  # Discord API limit is 100
        before: str | None = None,
        after: str | None = None,
        around: str | None = None,
    ) -> dict | list | None:
        """Get messages from a channel"""
        params: dict[str, str | int] = {"limit": limit}
        if before:
            params["before"] = before
        if after:
            params["after"] = after
        if around:
            params["around"] = around
        return await self._make_request("GET", f"/channels/{channel_id}/messages", params=params)

    async def send_message(self, channel_id: str, content: str) -> dict | list | None:
        """Send a message to a channel"""
        data = {"content": content}
        return await self._make_request("POST", f"/channels/{channel_id}/messages", json=data)

    async def get_channel_info(self, channel_id: str) -> dict | list | None:
        """Get channel information"""
        return await self._make_request("GET", f"/channels/{channel_id}")

    async def get_guild_info(self, guild_id: str) -> dict | list | None:
        """Get guild (server) information"""
        return await self._make_request("GET", f"/guilds/{guild_id}")


# Create FastMCP server
app = FastMCP("Discord MCP Server", instructions=__doc__)


@app.tool()
async def get_current_user():
    """Get information about the current Discord user"""
    async with DiscordAPI(DISCORD_TOKEN) as api:
        result = await api.get_current_user()
        return "[[ no return ]]" if result is None else dump(result)


@app.tool()
async def list_user_guilds():
    """List all Discord servers (guilds) the user is a member of"""
    async with DiscordAPI(DISCORD_TOKEN) as api:
        guilds = await api.get_user_guilds()
        if guilds is None:
            return "[[ no return ]]"
        result = [{"id": g["id"], "name": g["name"], "owner": g["owner"]} for g in guilds]
        return dump(result)


@app.tool()
async def get_guild_info(guild_id: str):
    """Get detailed information about a Discord server (guild)"""
    async with DiscordAPI(DISCORD_TOKEN) as api:
        result = await api.get_guild_info(guild_id)
        return "[[ no return ]]" if result is None else dump(result)


@app.tool()
async def list_guild_channels(guild_id: str):
    """List all channels in a Discord server (guild)"""
    async with DiscordAPI(DISCORD_TOKEN) as api:
        channels = await api.get_guild_channels(guild_id)
        if channels is None:
            return "[[ no return ]]"
        result = [{"id": c["id"], "name": c["name"], "type": c["type"]} for c in channels]
        return dump(result)


@app.tool()
async def get_channel_info(channel_id: str):
    """Get information about a Discord channel"""
    async with DiscordAPI(DISCORD_TOKEN) as api:
        result = await api.get_channel_info(channel_id)
        return "[[ no return ]]" if result is None else dump(result)


@app.tool()
async def read_channel_messages(
    channel_id: str,
    limit: int = 50,
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
        messages = await api.get_channel_messages(channel_id, limit, before, after, around)
        if messages is None:
            return "[[ no return ]]"
        result = [
            {
                "id": m["id"],
                "content": m["content"],
                "author": {
                    "id": m["author"]["id"],
                    "username": m["author"]["username"],
                    "global_name": m["author"].get("global_name"),
                },
                "timestamp": m["timestamp"],
            }
            for m in messages
        ]
        return dump(result)


@app.tool()
async def send_channel_message(channel_id: str, content: str, ctx: Context):
    """Send a message to a Discord channel"""
    async with DiscordAPI(DISCORD_TOKEN) as api:
        channel_info = await api.get_channel_info(channel_id)
        if channel_info is None or not isinstance(channel_info, dict):
            return dump({"status": "error", "message": "Failed to retrieve channel information"})

        channel_name = channel_info.get("name", f"Channel {channel_id}")
        guild_id = channel_info.get("guild_id")
        channel_type = channel_info.get("type", 0)

        # Get guild name if it's a guild channel
        guild_name = ""
        if guild_id:
            guild_info = await api.get_guild_info(guild_id)
            if guild_info and isinstance(guild_info, dict):
                guild_name = guild_info.get("name", "")

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

        assert confirm_result.action == "accept", confirm_result

        message = await api.send_message(channel_id, content)
        if message is None or not isinstance(message, dict):
            return "[[ no return ]]"
        result = {
            "id": message["id"],
            "content": message["content"],
            "channel_id": message["channel_id"],
            "timestamp": message["timestamp"],
        }
        return dump(result)


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
