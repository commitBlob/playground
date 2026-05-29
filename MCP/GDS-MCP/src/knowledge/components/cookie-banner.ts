import { ComponentDefinition } from "../types.js";

export const cookieBanner: ComponentDefinition = {
  slug: "cookie-banner",
  name: "Cookie banner",
  category: "layout",
  description:
    "Ask users about cookie preferences and confirm their choice, meeting UK cookie compliance requirements.",
  useCases: [
    "cookie consent",
    "ask about analytics cookies",
    "cookie preferences banner",
    "GDPR cookie notice",
  ],
  antiUseCases: [
    "Don't show the banner to users who have already made their choice.",
  ],
  relatedComponents: ["notification-banner"],
  template: {
    baseMarkup: `<div class="govuk-cookie-banner" data-nosnippet role="region" aria-label="Cookies on {{serviceName}}">
  <div class="govuk-cookie-banner__message govuk-width-container">
    <div class="govuk-grid-row">
      <div class="govuk-grid-column-two-thirds">
        <h2 class="govuk-cookie-banner__heading govuk-heading-m">Cookies on {{serviceName}}</h2>
        <div class="govuk-cookie-banner__content">
          <p class="govuk-body">We use some essential cookies to make this service work.</p>
          <p class="govuk-body">We'd also like to use analytics cookies so we can understand how you use the service and make improvements.</p>
        </div>
      </div>
    </div>
    <div class="govuk-button-group">
      <button type="button" class="govuk-button" data-module="govuk-button">
        Accept analytics cookies
      </button>
      <button type="button" class="govuk-button" data-module="govuk-button">
        Reject analytics cookies
      </button>
      <a class="govuk-link" href="/cookies">View cookies</a>
    </div>
  </div>
</div>`,
    slots: [
      { name: "serviceName", required: true, type: "text", description: "Your service name" },
    ],
    variants: [
      { name: "confirmation", description: "Confirmation message after user makes a choice" },
    ],
  },
  accessibilityRules: [
    {
      id: "cookie-banner-region",
      severity: "error",
      description: "Cookie banner must have role=\"region\" and aria-label",
      check: (html) => {
        if (!/govuk-cookie-banner/.test(html)) return { passed: true, message: "No cookie banner present." };
        if (!/role=["']region["']/.test(html)) {
          return { passed: false, message: "Cookie banner missing role=\"region\".", suggestion: "Add role=\"region\" to the cookie banner container." };
        }
        if (!/aria-label/.test(html)) {
          return { passed: false, message: "Cookie banner missing aria-label.", suggestion: "Add aria-label=\"Cookies on [service name]\" to the banner." };
        }
        return { passed: true, message: "Cookie banner has correct ARIA attributes." };
      },
    },
  ],
};
