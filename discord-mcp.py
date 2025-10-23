#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "aiohttp~=3.13.1",
#     "dotenv-pth",
#     "fastmcp~=2.13.0rc2",
#     "logfire[aiohttp-client]~=4.14.1",
# ]
# ///

"""
Discord MCP Server using FastMCP
A Model Context Protocol server for Discord API integration using user access tokens.
"""

from contextlib import suppress
from os import getenv

import aiohttp
from fastmcp import FastMCP

# Discord API configuration
DISCORD_API_BASE = "https://discord.com/api/v9"
DISCORD_TOKEN: str = getenv("DISCORD_TOKEN")  # type: ignore
assert DISCORD_TOKEN is not None, "Please set the DISCORD_TOKEN environment variable."


class DiscordAPI:
    def __init__(self, token: str):
        self.token = token
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": self.token,
                "Content-Type": "application/json",
                "User-Agent": "DiscordMCP/1.0",
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

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
        limit: int = 50,
        before: str | None = None,
        after: str | None = None,
        around: str | None = None,
    ) -> dict | list | None:
        """Get messages from a channel"""
        params: dict[str, str | int] = {"limit": min(limit, 100)}  # Discord API limit is 100
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
app = FastMCP("Discord MCP Server", __doc__)


@app.tool
async def get_current_user() -> dict | list | None:
    """Get information about the current Discord user"""
    async with DiscordAPI(DISCORD_TOKEN) as api:
        return await api.get_current_user()


@app.tool
async def list_user_guilds() -> dict | list | None:
    """List all Discord servers (guilds) the user is a member of"""
    async with DiscordAPI(DISCORD_TOKEN) as api:
        guilds = await api.get_user_guilds()
        if guilds is None:
            return None
        return [{"id": g["id"], "name": g["name"], "owner": g["owner"]} for g in guilds]


@app.tool
async def get_guild_info(guild_id: str) -> dict | list | None:
    """Get detailed information about a Discord server (guild)"""
    async with DiscordAPI(DISCORD_TOKEN) as api:
        return await api.get_guild_info(guild_id)


@app.tool
async def list_guild_channels(guild_id: str) -> dict | list | None:
    """List all channels in a Discord server (guild)"""
    async with DiscordAPI(DISCORD_TOKEN) as api:
        channels = await api.get_guild_channels(guild_id)
        if channels is None:
            return None
        return [{"id": c["id"], "name": c["name"], "type": c["type"]} for c in channels]


@app.tool
async def get_channel_info(channel_id: str) -> dict | list | None:
    """Get information about a Discord channel"""
    async with DiscordAPI(DISCORD_TOKEN) as api:
        return await api.get_channel_info(channel_id)


@app.tool
async def read_channel_messages(
    channel_id: str,
    limit: int = 50,
    before: str | None = None,
    after: str | None = None,
    around: str | None = None,
) -> dict | list | None:
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
            return None
        return [
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


@app.tool
async def send_channel_message(channel_id: str, content: str) -> dict | list | None:
    """Send a message to a Discord channel"""
    async with DiscordAPI(DISCORD_TOKEN) as api:
        message = await api.send_message(channel_id, content)
        if message is None or not isinstance(message, dict):
            return None
        return {
            "id": message["id"],
            "content": message["content"],
            "channel_id": message["channel_id"],
            "timestamp": message["timestamp"],
        }


if LOGFIRE_TOKEN := getenv("LOGFIRE_TOKEN"):
    from threading import Thread
    from time import sleep

    def worker():
        sleep(0.5)
        import logfire

        logfire.configure(scrubbing=False, token=LOGFIRE_TOKEN)
        logfire.instrument_mcp()
        logfire.instrument_aiohttp_client()

    Thread(target=worker, daemon=True).start()

if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        app.run("stdio", show_banner=False)
