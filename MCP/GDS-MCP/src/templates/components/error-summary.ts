import { GenerateOptions } from "../../knowledge/types.js";

/**
 * items[].value is used as a URL fragment (`href="#${value}"`). It must be a
 * valid HTML id that matches a form field id on the same page
 * (e.g. "full-name", "date-of-birth-day"). Do not pass arbitrary user-provided
 * strings here — values are HTML-escaped but the resulting fragment URL would
 * be nonsensical for anything other than a simple field id.
 */
export function renderErrorSummary(options: GenerateOptions): string {
  const items = options.items || [];

  const errorLinks = items
    .map((item) => `<li><a href="#${item.value}">${item.text}</a></li>`)
    .join("\n");

  return `<div class="govuk-error-summary" data-module="govuk-error-summary">
  <div role="alert">
    <h2 class="govuk-error-summary__title">
      There is a problem
    </h2>
    <div class="govuk-error-summary__body">
      <ul class="govuk-list govuk-error-summary__list">
${errorLinks}
      </ul>
    </div>
  </div>
</div>`;
}
