import { GenerateOptions } from "../../knowledge/types.js";

export function renderSummaryList(options: GenerateOptions): string {
  const items = options.items || [];

  const rowsHtml = items
    .map((item) => {
      let row = `  <div class="govuk-summary-list__row">\n`;
      row += `    <dt class="govuk-summary-list__key">\n      ${item.text}\n    </dt>\n`;
      row += `    <dd class="govuk-summary-list__value">\n      ${item.value}\n    </dd>\n`;
      if (item.hint) {
        row += `    <dd class="govuk-summary-list__actions">\n      <a class="govuk-link" href="${item.hint}">\n        Change<span class="govuk-visually-hidden"> ${item.text.toLowerCase()}</span>\n      </a>\n    </dd>\n`;
      }
      row += `  </div>`;
      return row;
    })
    .join("\n");

  return `<dl class="govuk-summary-list">
${rowsHtml}
</dl>`;
}
