import { GenerateOptions } from "../../knowledge/types.js";

export function renderBreadcrumbs(options: GenerateOptions): string {
  const items = options.items || [];

  const itemsHtml = items
    .map((item, i) => {
      if (i === items.length - 1) {
        return `    <li class="govuk-breadcrumbs__list-item" aria-current="page">${item.text}</li>`;
      }
      return `    <li class="govuk-breadcrumbs__list-item">\n      <a class="govuk-breadcrumbs__link" href="${item.value}">${item.text}</a>\n    </li>`;
    })
    .join("\n");

  return `<nav class="govuk-breadcrumbs" aria-label="Breadcrumb">
  <ol class="govuk-breadcrumbs__list">
${itemsHtml}
  </ol>
</nav>`;
}
