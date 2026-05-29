import { getComponent } from "../knowledge/index.js";
import { getAllComponents } from "../knowledge/components/index.js";
import { GenerateOptions } from "../knowledge/types.js";
import { renderComponent } from "../templates/index.js";
import { escapeHtml } from "../templates/helpers.js";

const UNSAFE_PROTOCOL_PATTERN = /^(?:\s*|[^\S\r\n]*)?(?:javascript|data|vbscript)\s*:/i;
// Fields that hold raw HTML content (e.g. fieldset's inner form fields).
// These are intentionally NOT HTML-escaped; callers are responsible for
// ensuring these values contain safe, trusted HTML.
const RAW_HTML_OPTION_KEYS = new Set(["content"]);

// Fields whose values are placed directly into href/src/action attributes.
// Note: items[].value is intentionally excluded — it has dual semantics (URL in
// breadcrumbs/task-list, fragment ID in error-summary, form value in checkboxes/
// radios/select). Unsafe protocols in items[].value when used as href are caught
// by the post-render assertNoUnsafeProtocolsInMarkup() check instead.
const URL_LIKE_OPTION_KEYS = new Set([
  "href",
  "serviceUrl",
  "prevHref",
  "nextHref",
  "feedbackUrl",
]);

type SanitizedGenerateOptions = GenerateOptions & {
  readonly __sanitizedGenerateOptionsBrand: unique symbol;
};

function hasUnsafeProtocol(value: string): boolean {
  return UNSAFE_PROTOCOL_PATTERN.test(value);
}

function assertSafeUrl(value: string, fieldName: string): void {
  if (hasUnsafeProtocol(value)) {
    throw new Error(
      `Unsafe URL protocol in options.${fieldName}. Blocked protocols: javascript:, data:, vbscript:`
    );
  }
}

function sanitizeUnknown(value: unknown, keyPath = "options"): unknown {
  if (typeof value === "string") {
    const key = keyPath.split(".").at(-1) || "";
    if (RAW_HTML_OPTION_KEYS.has(key)) {
      // Raw HTML content: skip escaping but still validate for unsafe protocols.
      if (hasUnsafeProtocol(value)) {
        throw new Error(
          `Unsafe URL protocol in options.${keyPath.replace(/^options\./, "")}. Blocked protocols: javascript:, data:, vbscript:`
        );
      }
      return value;
    }
    if (URL_LIKE_OPTION_KEYS.has(key)) {
      assertSafeUrl(value, keyPath.replace(/^options\./, ""));
    }
    return escapeHtml(value);
  }

  if (Array.isArray(value)) {
    return value.map((item, index) => sanitizeUnknown(item, `${keyPath}[${index}]`));
  }

  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, nestedValue]) => [
        key,
        sanitizeUnknown(nestedValue, `${keyPath}.${key}`),
      ])
    );
  }

  return value;
}

/**
 * Sanitisation contract:
 * - All string values are HTML-escaped before entering templates.
 * - URL-like option fields are protocol-validated and reject unsafe schemes.
 * - Templates should assume options are pre-sanitised by this boundary.
 */
function sanitizeGenerateOptions(options: GenerateOptions): SanitizedGenerateOptions {
  return sanitizeUnknown(options) as SanitizedGenerateOptions;
}

function assertNoUnsafeProtocolsInMarkup(markup: string): void {
  const attrRegex = /(href|src|action)\s*=\s*(["'])(.*?)\2/gi;
  for (const match of markup.matchAll(attrRegex)) {
    const [, attrName, , attrValue] = match;
    if (hasUnsafeProtocol(attrValue)) {
      throw new Error(
        `Generated markup contains unsafe ${attrName} value "${attrValue}". Blocked protocols: javascript:, data:, vbscript:`
      );
    }
  }
}

export function handleGenerateMarkup(args: {
  component: string;
  options?: GenerateOptions;
}): { content: Array<{ type: "text"; text: string }>; isError?: boolean } {
  const component = getComponent(args.component);

  if (!component) {
    const available = getAllComponents()
      .map((c) => c.slug)
      .join(", ");
    return {
      content: [
        {
          type: "text",
          text: `Component "${args.component}" not found.\n\nAvailable components: ${available}\n\nUse the suggest_component tool to find the right component for your use case.`,
        },
      ],
      isError: true,
    };
  }

  let options: SanitizedGenerateOptions;
  try {
    options = sanitizeGenerateOptions(args.options || {});
  } catch (error) {
    return {
      content: [
        {
          type: "text",
          text:
            error instanceof Error
              ? error.message
              : "Invalid options provided.",
        },
      ],
      isError: true,
    };
  }

  let markup: string;
  try {
    markup = renderComponent(component, options);
    assertNoUnsafeProtocolsInMarkup(markup);
  } catch (error) {
    return {
      content: [
        {
          type: "text",
          text:
            error instanceof Error
              ? error.message
              : "Failed to generate markup.",
        },
      ],
      isError: true,
    };
  }

  let output = `\`\`\`html\n${markup}\n\`\`\`\n`;

  const notes: string[] = [];
  if (options.errorMessage) {
    notes.push(
      "Error state: includes visually hidden 'Error:' prefix and aria-describedby linking"
    );
  }
  if (options.hint) {
    notes.push("Hint: linked to input via aria-describedby for screen readers");
  }
  if (!options.label) {
    notes.push(
      "Warning: No label text provided — replace 'Label text' with your actual label"
    );
  }

  if (notes.length > 0) {
    output += "\n### Accessibility notes\n\n";
    for (const note of notes) {
      output += `- ${note}\n`;
    }
  }

  return { content: [{ type: "text", text: output }] };
}
