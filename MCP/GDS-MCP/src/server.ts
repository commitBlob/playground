import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { handleSuggestComponent } from "./tools/suggest-component.js";
import { handleGenerateMarkup } from "./tools/generate-markup.js";
import { handleReviewHtml } from "./tools/review-html.js";

function safeStringify(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function truncate(text: string, max = 8000): string {
  if (text.length <= max) return text;
  return `${text.slice(0, max)}\n... [truncated ${text.length - max} chars]`;
}

async function withToolLogging<TArgs, TResult>(
  toolName: string,
  args: TArgs,
  handler: (args: TArgs) => TResult | Promise<TResult>
): Promise<TResult> {
  const start = Date.now();
  console.error(`[MCP] ${toolName} request:\n${truncate(safeStringify(args))}`);

  try {
    const result = await handler(args);
    const durationMs = Date.now() - start;
    console.error(
      `[MCP] ${toolName} response (${durationMs}ms):\n${truncate(safeStringify(result))}`
    );
    return result;
  } catch (error) {
    const durationMs = Date.now() - start;
    console.error(
      `[MCP] ${toolName} error (${durationMs}ms): ${
        error instanceof Error ? error.stack || error.message : String(error)
      }`
    );
    throw error;
  }
}

export function createServer(): McpServer {
  const server = new McpServer({
    name: "gds-components",
    version: "1.0.0",
  });

  server.tool(
    "suggest_component",
    "Suggest the right GOV.UK Design System component for a given use case. Describe what users need to do and get a recommendation with rationale, alternatives, and quick-start markup.",
    {
      useCase: z
        .string()
        .describe(
          "Description of what users need to do, e.g. 'users need to enter their name' or 'select multiple options from a list'"
        ),
      context: z
        .string()
        .optional()
        .describe(
          "Additional context like form type, page purpose, or user journey stage"
        ),
    },
    async (args) =>
      withToolLogging("suggest_component", args, (requestArgs) =>
        handleSuggestComponent(requestArgs)
      )
  );

  server.tool(
    "generate_markup",
    "Generate accessible GOV.UK Frontend HTML markup for a specific component. Provide the component slug and configuration options to get production-ready HTML with accessibility attributes.",
    {
      component: z
        .string()
        .describe(
          "Component slug, e.g. 'text-input', 'checkboxes', 'radios', 'error-summary'"
        ),
      options: z
        .object({
          label: z.string().optional().describe("The visible label text"),
          hint: z
            .string()
            .optional()
            .describe("Hint text displayed below the label"),
          errorMessage: z
            .string()
            .optional()
            .describe("Error message for validation failures"),
          id: z
            .string()
            .optional()
            .describe("The id attribute for the element"),
          name: z
            .string()
            .optional()
            .describe("The name attribute for form submission"),
          classes: z.string().optional().describe("Additional CSS classes"),
          variant: z
            .string()
            .optional()
            .describe("Named variant like 'small', 'inverse', 'start'"),
          width: z
            .string()
            .optional()
            .describe(
              "Width: numeric (2-20 characters) or named (full, three-quarters, two-thirds, one-half)"
            ),
          inputType: z
            .string()
            .optional()
            .describe("HTML input type: text, email, tel, number, etc."),
          items: z
            .array(
              z.object({
                text: z.string().describe("Display text for the item"),
                value: z.string().describe("Value for form submission"),
                hint: z.string().optional().describe("Hint text for this item"),
                checked: z.boolean().optional().describe("Pre-selected state"),
                selected: z.boolean().optional().describe("Selected state (for select)"),
                conditional: z
                  .string()
                  .optional()
                  .describe("HTML to show conditionally when selected"),
              })
            )
            .optional()
            .describe("Items for checkboxes, radios, or select components"),
        })
        .optional()
        .describe("Configuration options for the component"),
    },
    async (args) =>
      withToolLogging("generate_markup", args, (requestArgs) =>
        handleGenerateMarkup({
          component: requestArgs.component,
          options: requestArgs.options,
        })
      )
  );

  server.tool(
    "review_html",
    "Review HTML against GOV.UK Design System rules and accessibility patterns. Returns errors (must fix), warnings (should fix), and passed checks with fix suggestions.",
    {
      html: z
        .string()
        .describe("The HTML markup to review"),
      component: z
        .string()
        .optional()
        .describe(
          "If known, the specific component slug to check against (enables component-specific rules)"
        ),
      strictness: z
        .enum(["strict", "moderate", "lenient"])
        .optional()
        .describe(
          "Review strictness: strict (all rules), moderate (default), lenient (errors only)"
        ),
    },
    async (args) =>
      withToolLogging("review_html", args, (requestArgs) =>
        handleReviewHtml(requestArgs)
      )
  );

  return server;
}
