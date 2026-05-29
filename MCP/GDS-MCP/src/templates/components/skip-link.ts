import { GenerateOptions } from "../../knowledge/types.js";

export function renderSkipLink(options: GenerateOptions): string {
  const href = (options.href as string) || "#main-content";
  return `<a href="${href}" class="govuk-skip-link" data-module="govuk-skip-link">Skip to main content</a>`;
}
