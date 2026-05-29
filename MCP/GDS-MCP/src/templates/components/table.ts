import { GenerateOptions } from "../../knowledge/types.js";

export function renderTable(options: GenerateOptions): string {
  const caption = options.label || (options.caption as string) || "Table caption";
  const items = options.items || [];

  if (items.length === 0) {
    return `<table class="govuk-table">
  <caption class="govuk-table__caption govuk-table__caption--m">${caption}</caption>
  <thead class="govuk-table__head">
    <tr class="govuk-table__row">
      <th scope="col" class="govuk-table__header">Column 1</th>
      <th scope="col" class="govuk-table__header">Column 2</th>
    </tr>
  </thead>
  <tbody class="govuk-table__body">
    <tr class="govuk-table__row">
      <td class="govuk-table__cell">Data 1</td>
      <td class="govuk-table__cell">Data 2</td>
    </tr>
  </tbody>
</table>`;
  }

  const headers = items[0];
  const headCells = headers.text
    .split("|")
    .map((h: string) => `      <th scope="col" class="govuk-table__header">${h.trim()}</th>`)
    .join("\n");

  const bodyRows = items
    .slice(1)
    .map((item) => {
      const cells = item.value
        .split("|")
        .map((c: string) => `      <td class="govuk-table__cell">${c.trim()}</td>`)
        .join("\n");
      return `    <tr class="govuk-table__row">\n${cells}\n    </tr>`;
    })
    .join("\n");

  return `<table class="govuk-table">
  <caption class="govuk-table__caption govuk-table__caption--m">${caption}</caption>
  <thead class="govuk-table__head">
    <tr class="govuk-table__row">
${headCells}
    </tr>
  </thead>
  <tbody class="govuk-table__body">
${bodyRows}
  </tbody>
</table>`;
}
