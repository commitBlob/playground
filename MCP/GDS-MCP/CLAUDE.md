# GDS Components MCP Server

Read-only MCP server that helps coding agents use GOV.UK Design System components correctly.

## Quick Start

```bash
npm install
npm test          # vitest — 104 tests
npm run dev       # start server on stdio (tsx)
npm run build     # compile to dist/
npm start         # run compiled server
```

## Architecture

Three tools (`suggest_component`, `generate_markup`, `review_html`) over a deep knowledge module containing 34 GOV.UK Design System components.

```
src/
├── index.ts              # STDIO transport entry point
├── server.ts             # McpServer + tool registrations
├── knowledge/            # Deep module: components + rules
│   ├── types.ts          # Shared type definitions
│   ├── index.ts          # Public API (findByUseCase, getComponent, runAllRules)
│   ├── components/       # One file per component (34 files)
│   └── rules/            # Accessibility + design-pattern rules
├── tools/                # Thin tool handlers
└── templates/            # HTML rendering
```

## Adding a Component

1. Create `src/knowledge/components/[slug].ts` exporting a `ComponentDefinition`
2. Import and add to the array in `src/knowledge/components/index.ts`
3. Add a rendering function in `src/templates/index.ts`
4. Add the slug case to the switch in `renderComponent()`
5. Write tests

## Key Decisions

- GOV.UK Frontend v5.x markup
- No external dependencies at runtime (only MCP SDK + Zod)
- No HTML parser — rules use regex/string matching (GOV.UK markup is predictable)
- STDIO transport only (local use with Claude Code/Desktop)
- All component knowledge baked in (no API calls)

## Connecting to Claude Code

Add to `.claude/settings.local.json`:
```json
{
  "mcpServers": {
    "gds-components": {
      "command": "npx",
      "args": ["tsx", "/path/to/GDS-MCP/src/index.ts"]
    }
  }
}
```
