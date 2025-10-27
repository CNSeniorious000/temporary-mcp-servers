# temporary-mcp-servers

A collection of MCP servers for personal use cases. These are rough implementations - for polished, well-designed MCP servers, check out [refined-mcp-servers](https://github.com/promplate/refined-mcp-servers).

## Servers

### Discord MCP Server

Access Discord API via MCP. Requires `DISCORD_TOKEN` environment variable.

```sh
uv run https://raw.githubusercontent.com/CNSeniorious000/temporary-mcp-servers/HEAD/discord-mcp.py
```

### IPython MCP Server

Execute Python code in persistent IPython sessions via MCP.

```sh
uv run https://raw.githubusercontent.com/CNSeniorious000/temporary-mcp-servers/HEAD/ipython-mcp.py
```

## Requirements

- uv package manager
- For Discord server: Discord user token in `DISCORD_TOKEN` env var
- Optional: Logfire token in `LOGFIRE_TOKEN` env var for observability
