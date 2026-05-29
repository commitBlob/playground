import { GenerateOptions } from "../../knowledge/types.js";
import { toSlug } from "../helpers.js";
import { buildAriaDescribedBy, buildErrorHtml, buildHintHtml } from "./_form-helpers.js";

export function renderFileUpload(options: GenerateOptions): string {
  const id = options.id || toSlug(options.label || "file-upload");
  const name = options.name || id;
  const hasError = !!options.errorMessage;
  const hasHint = !!options.hint;

  const ariaIds: string[] = [];
  if (hasHint) ariaIds.push(`${id}-hint`);
  if (hasError) ariaIds.push(`${id}-error`);

  return `<div class="govuk-form-group${hasError ? " govuk-form-group--error" : ""}">
  <label class="govuk-label" for="${id}">
    ${options.label || "Upload a file"}
  </label>
${buildHintHtml(id, options.hint)}${buildErrorHtml(id, options.errorMessage)}  <input class="govuk-file-upload${hasError ? " govuk-file-upload--error" : ""}" id="${id}" name="${name}" type="file"${buildAriaDescribedBy(ariaIds)}>
</div>`;
}
