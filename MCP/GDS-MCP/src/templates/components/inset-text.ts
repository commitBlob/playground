import { GenerateOptions } from "../../knowledge/types.js";

export function renderInsetText(options: GenerateOptions): string {
  const content = (options.content as string) || options.label || "Inset text content.";

  return `<div class="govuk-inset-text">
  ${content}
</div>`;
}
