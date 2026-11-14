#!/bin/zsh
# Prints the Authorization header for the Excalidraw+ MCP server (see .mcp.json).
# Reads EXCALIDRAW_API_KEY from ~/.zshrc so the key never lives in this repo.
[[ -z "$EXCALIDRAW_API_KEY" ]] && source ~/.zshrc >/dev/null 2>&1
printf '{"Authorization": "Bearer %s"}\n' "$EXCALIDRAW_API_KEY"
