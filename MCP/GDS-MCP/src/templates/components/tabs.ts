import { GenerateOptions } from "../../knowledge/types.js";
import { toSlug } from "../helpers.js";

export function renderTabs(options: GenerateOptions): string {
  const title = (options.title as string) || "Contents";
  const items = options.items || [];

  const tabItems = items
    .map((item, i) => {
      const tabId = toSlug(item.text);
      const cls = i === 0 ? "govuk-tabs__list-item govuk-tabs__list-item--selected" : "govuk-tabs__list-item";
      return `    <li class="${cls}">\n      <a class="govuk-tabs__tab" href="#${tabId}">\n        ${item.text}\n      </a>\n    </li>`;
    })
    .join("\n");

  const tabPanels = items
    .map((item, i) => {
      const tabId = toSlug(item.text);
      const hiddenAttr = i === 0 ? "" : " hidden";
      return `  <div class="govuk-tabs__panel${i > 0 ? " govuk-tabs__panel--hidden" : ""}" id="${tabId}"${hiddenAttr}>\n    <p class="govuk-body">${item.value}</p>\n  </div>`;
    })
    .join("\n");

  return `<div class="govuk-tabs" data-module="govuk-tabs">
  <h2 class="govuk-tabs__title">
    ${title}
  </h2>
  <ul class="govuk-tabs__list">
${tabItems}
  </ul>
${tabPanels}
</div>`;
}
