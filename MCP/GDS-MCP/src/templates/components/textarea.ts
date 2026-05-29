import { GenerateOptions } from "../../knowledge/types.js";
import { toSlug } from "../helpers.js";
import { buildAriaDescribedBy, buildErrorHtml, buildHintHtml } from "./_form-helpers.js";

export function renderTextarea(options: GenerateOptions): string {
  const id = options.id || toSlug(options.label || "more-detail");
  const name = options.name || id;
  const rows = (options.rows as string) || "5";
  const hasError = !!options.errorMessage;
  const hasHint = !!options.hint;

  const ariaIds: string[] = [];
  if (hasHint) ariaIds.push(`${id}-hint`);
  if (hasError) ariaIds.push(`${id}-error`);

  return `<div class="govuk-form-group${hasError ? " govuk-form-group--error" : ""}">
  <label class="govuk-label" for="${id}">
    ${options.label || "Label text"}
  </label>
${buildHintHtml(id, options.hint)}${buildErrorHtml(id, options.errorMessage)}  <textarea class="govuk-textarea${hasError ? " govuk-textarea--error" : ""}" id="${id}" name="${name}" rows="${rows}"${buildAriaDescribedBy(ariaIds)}></textarea>
</div>`;
}
