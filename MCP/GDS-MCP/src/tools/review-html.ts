import {
  getComponent,
  runAllRules,
  runComponentRules,
  ReviewResult,
} from "../knowledge/index.js";

export function handleReviewHtml(args: {
  html: string;
  component?: string;
  strictness?: "strict" | "moderate" | "lenient";
}): { content: Array<{ type: "text"; text: string }> } {
  const MAX_HTML_BYTES = 1_000_000; // 1 MB
  if (args.html.length > MAX_HTML_BYTES) {
    return {
      content: [
        {
          type: "text",
          text: `HTML input exceeds maximum size of ${MAX_HTML_BYTES.toLocaleString()} characters. Provide a smaller snippet.`,
        },
      ],
    };
  }

  const strictness = args.strictness || "moderate";
  const results: ReviewResult[] = [];

  results.push(...runAllRules(args.html, strictness));

  if (args.component) {
    const comp = getComponent(args.component);
    if (comp && comp.accessibilityRules.length > 0) {
      results.push(...runComponentRules(args.html, comp.accessibilityRules));
    }
  }

  const deduped = deduplicateResults(results);
  const errors = deduped.filter((r) => !r.result.passed && r.severity === "error");
  const warnings = deduped.filter(
    (r) => !r.result.passed && r.severity === "warning"
  );
  const passed = deduped.filter((r) => r.result.passed);

  let output = "## Review Results\n\n";

  if (errors.length === 0 && warnings.length === 0) {
    output += "All checks passed. The HTML follows GOV.UK Design System patterns.\n\n";
  }

  if (errors.length > 0) {
    output += "### Errors (must fix)\n\n";
    for (const e of errors) {
      output += `1. **${e.description}**\n`;
      output += `   ${e.result.message}\n`;
      if (e.result.suggestion) {
        output += `   _Fix:_ ${e.result.suggestion}\n`;
      }
      output += "\n";
    }
  }

  if (warnings.length > 0) {
    output += "### Warnings (should fix)\n\n";
    for (const w of warnings) {
      output += `1. **${w.description}**\n`;
      output += `   ${w.result.message}\n`;
      if (w.result.suggestion) {
        output += `   _Fix:_ ${w.result.suggestion}\n`;
      }
      output += "\n";
    }
  }

  if (passed.length > 0) {
    output += "### Passed checks\n\n";
    for (const p of passed) {
      output += `- ${p.result.message}\n`;
    }
  }

  return { content: [{ type: "text", text: output }] };
}

// Deduplicate by ruleId so that rules which run via both global and
// component-specific paths (e.g. when component slug is provided) are
// only reported once, regardless of message text.
function deduplicateResults(results: ReviewResult[]): ReviewResult[] {
  const seen = new Set<string>();
  return results.filter((r) => {
    if (seen.has(r.ruleId)) return false;
    seen.add(r.ruleId);
    return true;
  });
}
