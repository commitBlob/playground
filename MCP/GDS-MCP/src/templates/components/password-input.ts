import { GenerateOptions } from "../../knowledge/types.js";
import { buildAriaDescribedBy, buildErrorHtml, buildHintHtml } from "./_form-helpers.js";

export function renderPasswordInput(options: GenerateOptions): string {
  const id = options.id || "password";
  const name = options.name || id;
  const autocomplete = (options.autocomplete as string) || "current-password";
  const hasError = !!options.errorMessage;
  const hasHint = !!options.hint;

  const ariaIds: string[] = [];
  if (hasHint) ariaIds.push(`${id}-hint`);
  if (hasError) ariaIds.push(`${id}-error`);

  return `<div class="govuk-form-group${hasError ? " govuk-form-group--error" : ""}">
  <label class="govuk-label" for="${id}">
    ${options.label || "Password"}
  </label>
${buildHintHtml(id, options.hint)}${buildErrorHtml(id, options.errorMessage)}  <div class="govuk-password-input" data-module="govuk-password-input">
    <input class="govuk-input govuk-password-input__input govuk-js-password-input-input${hasError ? " govuk-input--error" : ""}" id="${id}" name="${name}" type="password"${buildAriaDescribedBy(ariaIds)} spellcheck="false" autocomplete="${autocomplete}" autocapitalize="none">
    <button type="button" class="govuk-button govuk-button--secondary govuk-password-input__toggle govuk-js-password-input-toggle" data-module="govuk-button" aria-controls="${id}" aria-label="Show password" hidden>
      Show
    </button>
  </div>
</div>`;
}
