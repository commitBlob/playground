import { GenerateOptions } from "../../knowledge/types.js";
import { buildAriaDescribedBy, buildErrorHtml, buildHintHtml } from "./_form-helpers.js";

export function renderRadios(options: GenerateOptions): string {
  const id = options.id || "radios";
  const name = options.name || id;
  const legend = options.label || options.legend as string || "Select an option";
  const hasError = !!options.errorMessage;
  const hasHint = !!options.hint;
  const items = options.items || [];
  const isInline = options.variant === "inline";

  const ariaIds: string[] = [];
  if (hasHint) ariaIds.push(`${id}-hint`);
  if (hasError) ariaIds.push(`${id}-error`);

  const itemsHtml = items
    .map((item, i) => {
      const itemId = `${id}-${i + 1}`;
      let itemHtml = `      <div class="govuk-radios__item">\n`;
      itemHtml += `        <input class="govuk-radios__input" id="${itemId}" name="${name}" type="radio" value="${item.value}"${item.checked ? " checked" : ""}>\n`;
      itemHtml += `        <label class="govuk-label govuk-radios__label" for="${itemId}">\n          ${item.text}\n        </label>\n`;
      if (item.hint) {
        itemHtml += `        <div id="${itemId}-item-hint" class="govuk-hint govuk-radios__hint">\n          ${item.hint}\n        </div>\n`;
      }
      itemHtml += `      </div>`;
      return itemHtml;
    })
    .join("\n");

  const radiosClass = isInline ? "govuk-radios govuk-radios--inline" : "govuk-radios";

  return `<div class="govuk-form-group${hasError ? " govuk-form-group--error" : ""}">
  <fieldset class="govuk-fieldset"${buildAriaDescribedBy(ariaIds)}>
    <legend class="govuk-fieldset__legend">
      ${legend}
    </legend>
${buildHintHtml(id, options.hint)}${buildErrorHtml(id, options.errorMessage)}    <div class="${radiosClass}" data-module="govuk-radios">
${itemsHtml}
    </div>
  </fieldset>
</div>`;
}
