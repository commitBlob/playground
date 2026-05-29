# GDS Components MCP Server

A read-only [Model Context Protocol](https://modelcontextprotocol.io) server that helps coding agents use [GOV.UK Design System](https://design-system.service.gov.uk/) components correctly.

Embed GOV.UK accessibility patterns directly into your development workflow — get component suggestions, generate accessible HTML, and review markup against GOV.UK rules without leaving your editor.

## What it does

Three tools available to any MCP-compatible agent (Claude Code, Claude Desktop, etc.):

| Tool | Description |
|------|-------------|
| `suggest_component` | Describe your use case in plain English; get the right GOV.UK component with rationale and alternatives |
| `generate_markup` | Pass a component slug and options; get accessible GOV.UK Frontend v5.x HTML |
| `review_html` | Paste HTML; get errors, warnings, and accessibility checks against GOV.UK rules |

## Components covered

34 GOV.UK Design System components across all categories:

**Form inputs** — Text input, Textarea, Checkboxes, Radios, Character count, Date input, File upload, Password input, Select

**Navigation** — Accordion, Back link, Breadcrumbs, Pagination, Skip link, Service navigation, Tabs

**Content** — Details, Fieldset, Inset text, Summary list, Table, Task list, Tag, Warning text

**Messaging** — Error message, Error summary, Notification banner, Panel

**Layout & actions** — Button, Header, Footer, Cookie banner, Exit this page, Phase banner

## Quick start

```bash
npm install
npm test          # run test suite with vitest
npm run dev       # start server on stdio
npm run build     # compile to dist/
npm start         # run compiled server
```

## Connecting to Claude Code

Add to `.claude/settings.local.json` (or `settings.json` for all projects):

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

Or with the compiled build:

```json
{
  "mcpServers": {
    "gds-components": {
      "command": "node",
      "args": ["/path/to/GDS-MCP/dist/index.js"]
    }
  }
}
```

## Connecting to Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

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

## Example usage

**Suggest a component:**
> "I need users to select one option from a list of 4 choices"
→ Recommends Radios, explains when to use Checkboxes instead, provides quick-start HTML

**Generate markup:**
> generate_markup with `{ "component": "date-input", "label": "Date of birth", "hint": "For example, 27 3 1980" }`
→ Returns accessible HTML with fieldset, legend, three inputs, aria-describedby wired to the hint

**Review HTML:**
> review_html with your existing template HTML
→ Reports missing `for`/`id` associations, placeholder-only labels, missing error prefixes, incorrect class names

## Architecture

```
src/
├── index.ts              # STDIO transport entry point
├── server.ts             # McpServer + tool registrations (Zod schemas)
├── knowledge/            # Deep module: all component knowledge
│   ├── types.ts          # Shared type definitions
│   ├── index.ts          # Public API: findByUseCase, getComponent, runAllRules
│   ├── components/       # One file per component (34 files)
│   └── rules/            # Accessibility + design-pattern rule engine
├── tools/                # Thin handlers delegating to knowledge/
└── templates/            # HTML rendering (no template engine — string construction)
```

No HTML parser, no template engine, no API calls at runtime. Rules use regex/string matching — GOV.UK markup is predictable enough that this is reliable and fast.

## Adding a component

1. Create `src/knowledge/components/[slug].ts` exporting a `ComponentDefinition`
2. Import and add to the array in `src/knowledge/components/index.ts`
3. Create `src/templates/components/[slug].ts` exporting `render[ComponentName](options)`
4. Import the renderer in `src/templates/index.ts` and add the slug case to `renderComponent()`
5. Write tests in `tests/`

> **Note:** Steps 2 and 4 must stay in sync. If a component is registered in the knowledge index but has no renderer case, `generate_markup` will throw at runtime. Adding a test that calls `generate_markup` for the new component is sufficient to catch this.

## Security notes

### Input sanitisation

`generate_markup` applies two layers of protection before returning output:

1. **HTML escaping** — all string values in options are HTML-escaped (`&`, `<`, `>`, `"`) before entering any template renderer.
2. **URL protocol validation** — named URL option fields (`href`, `serviceUrl`, `prevHref`, `nextHref`, `feedbackUrl`) and the final rendered markup are checked for unsafe schemes (`javascript:`, `data:`, `vbscript:`). Requests containing unsafe schemes are rejected with an error.

Templates assume options are pre-sanitised by `handleGenerateMarkup()`. If you call renderers directly (bypassing the tool handler), sanitisation is your responsibility.

### Review rules

`review_html` includes a rule (`no-unsafe-url-protocols`) that flags unsafe URL schemes in `href`, `src`, and `action` attributes in submitted HTML.

All checks use regex/string matching rather than a full HTML parser — GOV.UK markup is sufficiently predictable for this approach to be reliable. Edge cases (unusual attribute ordering, multi-line attributes, HTML entities in URLs) may not be caught; treat `review_html` as a strong assistant, not a formal security audit.

## Design decisions

- **GOV.UK Frontend v5.x** — uses current class names and markup patterns
- **STDIO transport only** — intended for local use with Claude Code/Desktop
- **No external runtime dependencies** — only `@modelcontextprotocol/sdk` and `zod`
- **Deep module design** — three simple tool interfaces over a rich 34-component knowledge base
- **Keyword-weighted matching** — suggestion engine uses stop words, word-length weighting, negative matching (antiUseCases), and slug/name boosting to disambiguate close matches

## Requirements

- Node.js 18+
- TypeScript 5.4+ (dev only)
