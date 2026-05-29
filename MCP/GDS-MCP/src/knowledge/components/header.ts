import { ComponentDefinition } from "../types.js";

export const header: ComponentDefinition = {
  slug: "header",
  name: "Header",
  category: "layout",
  description:
    "The GOV.UK page header with crown logo, service name, and optional navigation links.",
  useCases: [
    "page header with GOV.UK branding",
    "service header",
    "top of page navigation",
    "site header with logo",
  ],
  antiUseCases: [
    "Don't modify the GOV.UK logo or crown — it must match the standard.",
    "Don't add too many navigation items — keep it to essential service-level links.",
  ],
  relatedComponents: ["footer", "service-navigation"],
  template: {
    baseMarkup: `<header class="govuk-header" data-module="govuk-header">
  <div class="govuk-header__container govuk-width-container">
    <div class="govuk-header__logo">
      <a href="/" class="govuk-header__link govuk-header__link--homepage">
        <span class="govuk-header__logotype">
          <span class="govuk-header__logotype-text">GOV.UK</span>
        </span>
      </a>
    </div>
    <div class="govuk-header__content">
      <a href="{{serviceUrl}}" class="govuk-header__link govuk-header__service-name">
        {{serviceName}}
      </a>
    </div>
  </div>
</header>`,
    slots: [
      { name: "serviceName", required: false, type: "text", description: "Your service name" },
      { name: "serviceUrl", required: false, type: "text", description: "URL for the service name link", default: "/" },
    ],
    variants: [
      { name: "with-navigation", description: "Header with navigation menu items" },
    ],
  },
  accessibilityRules: [],
};
