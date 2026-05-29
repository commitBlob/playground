import { GenerateOptions } from "../../knowledge/types.js";
import { buildAriaDescribedBy, buildErrorHtml, buildHintHtml } from "./_form-helpers.js";

export function renderCheckboxes(options: GenerateOptions): string {
  const id = options.id || "checkboxes";
  const name = options.name || id;
  const legend = options.label || options.legend as string || "Select options";
  const hasError = !!options.errorMessage;
  const hasHint = !!options.hint;
  const items = options.items || [];

  const ariaIds: string[] = [];
  if (hasHint) ariaIds.push(`${id}-hint`);
  if (hasError) ariaIds.push(`${id}-error`);

  const itemsHtml = items
    .map((item, i) => {
      const itemId = `${id}-${i + 1}`;
      let itemHtml = `      <div class="govuk-checkboxes__item">\n`;
      itemHtml += `        <input class="govuk-checkboxes__input" id="${itemId}" name="${name}" type="checkbox" value="${item.value}"${item.checked ? " checked" : ""}>\n`;
      itemHtml += `        <label class="govuk-label govuk-checkboxes__label" for="${itemId}">\n          ${item.text}\n        </label>\n`;
      if (item.hint) {
        itemHtml += `        <div id="${itemId}-item-hint" class="govuk-hint govuk-checkboxes__hint">\n          ${item.hint}\n        </div>\n`;
      }
      itemHtml += `      </div>`;
      return itemHtml;
    })
    .join("\n");

  return `<div class="govuk-form-group${hasError ? " govuk-form-group--error" : ""}">
  <fieldset class="govuk-fieldset"${buildAriaDescribedBy(ariaIds)}>
    <legend class="govuk-fieldset__legend">
      ${legend}
    </legend>
${buildHintHtml(id, options.hint)}${buildErrorHtml(id, options.errorMessage)}    <div class="govuk-checkboxes" data-module="govuk-checkboxes">
${itemsHtml}
    </div>
  </fieldset>
</div>`;
}
