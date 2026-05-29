import { GenerateOptions } from "../../knowledge/types.js";
import { toSlug } from "../helpers.js";
import { buildAriaDescribedBy, buildErrorHtml, buildHintHtml } from "./_form-helpers.js";

export function renderTextInput(options: GenerateOptions): string {
  const id = options.id || toSlug(options.label || "input-1");
  const name = options.name || id;
  const inputType = (options.inputType as string) || "text";
  const hasError = !!options.errorMessage;
  const hasHint = !!options.hint;

  const ariaIds: string[] = [];
  if (hasHint) ariaIds.push(`${id}-hint`);
  if (hasError) ariaIds.push(`${id}-error`);

  let widthClass = "";
  if (options.width) {
    const w = options.width as string;
    widthClass = /^\d+$/.test(w) ? ` govuk-input--width-${w}` : ` govuk-!-width-${w}`;
  }

  const extraClasses = options.classes ? ` ${options.classes}` : "";

  return `<div class="govuk-form-group${hasError ? " govuk-form-group--error" : ""}">
  <label class="govuk-label" for="${id}">
    ${options.label || "Label text"}
  </label>
${buildHintHtml(id, options.hint)}${buildErrorHtml(id, options.errorMessage)}  <input class="govuk-input${hasError ? " govuk-input--error" : ""}${widthClass}${extraClasses}" id="${id}" name="${name}" type="${inputType}"${buildAriaDescribedBy(ariaIds)}>
</div>`;
}
