/**
 * Pre-sanitisation contract: all string arguments passed to these helpers
 * must already be HTML-escaped. In normal usage this is guaranteed because
 * templates are only called via handleGenerateMarkup(), which runs
 * sanitizeGenerateOptions() before invoking any renderer.
 */

export function buildAriaDescribedBy(ids: string[]): string {
  return ids.length > 0 ? ` aria-describedby="${ids.join(" ")}"` : "";
}

/** @param hint - pre-HTML-escaped hint text */
export function buildHintHtml(id: string, hint?: string): string {
  if (!hint) return "";
  return `  <div id="${id}-hint" class="govuk-hint">\n    ${hint}\n  </div>\n`;
}

/** @param errorMessage - pre-HTML-escaped error message text */
export function buildErrorHtml(id: string, errorMessage?: string): string {
  if (!errorMessage) return "";
  return `  <p id="${id}-error" class="govuk-error-message">\n    <span class="govuk-visually-hidden">Error:</span> ${errorMessage}\n  </p>\n`;
}
