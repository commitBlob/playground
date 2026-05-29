import { GenerateOptions } from "../../knowledge/types.js";

export function renderErrorMessage(options: GenerateOptions): string {
  const id = options.id || "field";
  const message = options.label || (options.message as string) || "Error message";

  return `<p id="${id}-error" class="govuk-error-message">
  <span class="govuk-visually-hidden">Error:</span> ${message}
</p>`;
}
