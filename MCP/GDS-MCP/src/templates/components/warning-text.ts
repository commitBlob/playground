import { GenerateOptions } from "../../knowledge/types.js";

export function renderWarningText(options: GenerateOptions): string {
  const text = options.label || (options.text as string) || "Warning text";

  return `<div class="govuk-warning-text">
  <span class="govuk-warning-text__icon" aria-hidden="true">!</span>
  <strong class="govuk-warning-text__text">
    <span class="govuk-visually-hidden">Warning</span>
    ${text}
  </strong>
</div>`;
}
