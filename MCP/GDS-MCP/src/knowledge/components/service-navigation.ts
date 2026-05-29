import { ComponentDefinition } from "../types.js";

export const serviceNavigation: ComponentDefinition = {
  slug: "service-navigation",
  name: "Service navigation",
  category: "navigation",
  description:
    "A horizontal navigation bar for service-level links, typically placed below the header.",
  useCases: [
    "service level navigation",
    "horizontal nav links",
    "top navigation bar for service sections",
    "navigation between service areas",
  ],
  antiUseCases: [
    "Don't duplicate links that are already in the header.",
    "For page-level navigation within content, use Tabs or Accordion instead.",
  ],
  relatedComponents: ["header", "tabs"],
  template: {
    baseMarkup: `<nav class="govuk-service-navigation" data-module="govuk-service-navigation" aria-label="Service">
  <div class="govuk-width-container">
    <div class="govuk-service-navigation__container">
      <span class="govuk-service-navigation__service-name">
        <a href="/" class="govuk-service-navigation__link">
          {{serviceName}}
        </a>
      </span>
      <nav aria-label="Menu" class="govuk-service-navigation__wrapper">
        <ul class="govuk-service-navigation__list">
{{navItems}}
        </ul>
      </nav>
    </div>
  </div>
</nav>`,
    slots: [
      { name: "serviceName", required: true, type: "text", description: "The service name" },
      { name: "items", required: true, type: "array", description: "Navigation items with text and href" },
    ],
    variants: [],
  },
  accessibilityRules: [
    {
      id: "service-nav-aria-label",
      severity: "warning",
      description: "Service navigation should have aria-label",
      check: (html) => {
        if (!/govuk-service-navigation/.test(html)) return { passed: true, message: "No service navigation present." };
        if (!/aria-label/.test(html)) {
          return { passed: false, message: "Service navigation missing aria-label.", suggestion: "Add aria-label to the <nav> element." };
        }
        return { passed: true, message: "Service navigation has aria-label." };
      },
    },
  ],
};
