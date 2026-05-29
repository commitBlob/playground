import { GenerateOptions } from "../../knowledge/types.js";

export function renderServiceNavigation(options: GenerateOptions): string {
  const serviceName = (options.serviceName as string) || options.label || "Service";
  const items = options.items || [];

  const navItems = items
    .map((item) => `          <li class="govuk-service-navigation__item">\n            <a class="govuk-service-navigation__link" href="${item.value}">\n              ${item.text}\n            </a>\n          </li>`)
    .join("\n");

  return `<nav class="govuk-service-navigation" data-module="govuk-service-navigation" aria-label="Service">
  <div class="govuk-width-container">
    <div class="govuk-service-navigation__container">
      <span class="govuk-service-navigation__service-name">
        <a href="/" class="govuk-service-navigation__link">
          ${serviceName}
        </a>
      </span>
      <nav aria-label="Menu" class="govuk-service-navigation__wrapper">
        <ul class="govuk-service-navigation__list">
${navItems}
        </ul>
      </nav>
    </div>
  </div>
</nav>`;
}
