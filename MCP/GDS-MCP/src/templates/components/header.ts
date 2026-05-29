import { GenerateOptions } from "../../knowledge/types.js";

export function renderHeader(options: GenerateOptions): string {
  const serviceName = (options.serviceName as string) || options.label || "";
  const serviceUrl = (options.serviceUrl as string) || "/";

  let serviceHtml = "";
  if (serviceName) {
    serviceHtml = `\n    <div class="govuk-header__content">\n      <a href="${serviceUrl}" class="govuk-header__link govuk-header__service-name">\n        ${serviceName}\n      </a>\n    </div>`;
  }

  return `<header class="govuk-header" data-module="govuk-header">
  <div class="govuk-header__container govuk-width-container">
    <div class="govuk-header__logo">
      <a href="/" class="govuk-header__link govuk-header__link--homepage">
        <span class="govuk-header__logotype">
          <span class="govuk-header__logotype-text">GOV.UK</span>
        </span>
      </a>
    </div>${serviceHtml}
  </div>
</header>`;
}
