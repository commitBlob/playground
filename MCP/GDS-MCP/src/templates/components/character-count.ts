import { GenerateOptions } from "../../knowledge/types.js";
import { toSlug } from "../helpers.js";
import { buildAriaDescribedBy, buildErrorHtml, buildHintHtml } from "./_form-helpers.js";

export function renderCharacterCount(options: GenerateOptions): string {
  const id = options.id || toSlug(options.label || "with-hint");
  const name = options.name || id;
  const rows = (options.rows as string) || "5";
  const maxlength = (options.maxlength as string) || "200";
  const hasError = !!options.errorMessage;
  const hasHint = !!options.hint;

  const ariaIds: string[] = [];
  if (hasHint) ariaIds.push(`${id}-hint`);
  if (hasError) ariaIds.push(`${id}-error`);
  ariaIds.push(`${id}-info`);

  return `<div class="govuk-character-count" data-module="govuk-character-count" data-maxlength="${maxlength}">
  <div class="govuk-form-group${hasError ? " govuk-form-group--error" : ""}">
    <label class="govuk-label" for="${id}">
      ${options.label || "Label text"}
    </label>
${buildHintHtml(id, options.hint)}${buildErrorHtml(id, options.errorMessage)}    <textarea class="govuk-textarea govuk-js-character-count${hasError ? " govuk-textarea--error" : ""}" id="${id}" name="${name}" rows="${rows}"${buildAriaDescribedBy(ariaIds)}></textarea>
  </div>
  <div id="${id}-info" class="govuk-hint govuk-character-count__message">
    You can enter up to ${maxlength} characters
  </div>
</div>`;
}
