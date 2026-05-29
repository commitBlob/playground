import { GenerateOptions } from "../../knowledge/types.js";
import { buildAriaDescribedBy, buildErrorHtml, buildHintHtml } from "./_form-helpers.js";

export function renderDateInput(options: GenerateOptions): string {
  const id = options.id || "date";
  const name = options.name || id;
  const legend = options.label || options.legend as string || "Date";
  const hasError = !!options.errorMessage;
  const hasHint = !!options.hint;

  const ariaIds: string[] = [];
  if (hasHint) ariaIds.push(`${id}-hint`);
  if (hasError) ariaIds.push(`${id}-error`);

  return `<div class="govuk-form-group${hasError ? " govuk-form-group--error" : ""}">
  <fieldset class="govuk-fieldset" role="group"${buildAriaDescribedBy(ariaIds)}>
    <legend class="govuk-fieldset__legend">
      ${legend}
    </legend>
${buildHintHtml(id, options.hint)}${buildErrorHtml(id, options.errorMessage)}    <div class="govuk-date-input" id="${id}">
      <div class="govuk-date-input__item">
        <div class="govuk-form-group">
          <label class="govuk-label govuk-date-input__label" for="${id}-day">
            Day
          </label>
          <input class="govuk-input govuk-date-input__input govuk-input--width-2${hasError ? " govuk-input--error" : ""}" id="${id}-day" name="${name}-day" type="text" inputmode="numeric">
        </div>
      </div>
      <div class="govuk-date-input__item">
        <div class="govuk-form-group">
          <label class="govuk-label govuk-date-input__label" for="${id}-month">
            Month
          </label>
          <input class="govuk-input govuk-date-input__input govuk-input--width-2${hasError ? " govuk-input--error" : ""}" id="${id}-month" name="${name}-month" type="text" inputmode="numeric">
        </div>
      </div>
      <div class="govuk-date-input__item">
        <div class="govuk-form-group">
          <label class="govuk-label govuk-date-input__label" for="${id}-year">
            Year
          </label>
          <input class="govuk-input govuk-date-input__input govuk-input--width-4${hasError ? " govuk-input--error" : ""}" id="${id}-year" name="${name}-year" type="text" inputmode="numeric">
        </div>
      </div>
    </div>
  </fieldset>
</div>`;
}
