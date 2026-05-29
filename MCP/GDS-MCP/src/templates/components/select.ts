import { GenerateOptions } from "../../knowledge/types.js";
import { toSlug } from "../helpers.js";
import { buildAriaDescribedBy, buildErrorHtml, buildHintHtml } from "./_form-helpers.js";

export function renderSelect(options: GenerateOptions): string {
  const id = options.id || toSlug(options.label || "select-1");
  const name = options.name || id;
  const hasError = !!options.errorMessage;
  const hasHint = !!options.hint;
  const items = options.items || [];

  const ariaIds: string[] = [];
  if (hasHint) ariaIds.push(`${id}-hint`);
  if (hasError) ariaIds.push(`${id}-error`);

  const optionsHtml = items
    .map((item) => {
      const selected = item.selected ? " selected" : "";
      return `    <option value="${item.value}"${selected}>${item.text}</option>`;
    })
    .join("\n");

  return `<div class="govuk-form-group${hasError ? " govuk-form-group--error" : ""}">
  <label class="govuk-label" for="${id}">
    ${options.label || "Label text"}
  </label>
${buildHintHtml(id, options.hint)}${buildErrorHtml(id, options.errorMessage)}  <select class="govuk-select${hasError ? " govuk-select--error" : ""}" id="${id}" name="${name}"${buildAriaDescribedBy(ariaIds)}>
${optionsHtml}
  </select>
</div>`;
}
