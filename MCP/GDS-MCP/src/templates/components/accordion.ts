import { GenerateOptions } from "../../knowledge/types.js";

export function renderAccordion(options: GenerateOptions): string {
  const id = options.id || "accordion-default";
  const items = options.items || [];

  const sectionsHtml = items
    .map((item, i) => {
      const n = i + 1;
      return `  <div class="govuk-accordion__section">
    <div class="govuk-accordion__section-header">
      <h2 class="govuk-accordion__section-heading">
        <span class="govuk-accordion__section-button" id="${id}-heading-${n}">
          ${item.text}
        </span>
      </h2>
    </div>
    <div id="${id}-content-${n}" class="govuk-accordion__section-content">
      <p class="govuk-body">${item.value}</p>
    </div>
  </div>`;
    })
    .join("\n");

  return `<div class="govuk-accordion" data-module="govuk-accordion" id="${id}">
${sectionsHtml}
</div>`;
}
