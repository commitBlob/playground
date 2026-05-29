import { ComponentDefinition } from "../types.js";

export const backLink: ComponentDefinition = {
  slug: "back-link",
  name: "Back link",
  category: "navigation",
  description:
    "A link that takes users back to the previous page in a multi-page transaction.",
  useCases: [
    "go back to previous page",
    "back button",
    "navigate backwards in a flow",
    "return to previous step",
  ],
  antiUseCases: [
    "Don't use on the first page of a service — there's no previous page.",
    "Don't use alongside breadcrumbs — choose one navigation pattern.",
    "For non-linear journeys, use a link with specific text describing where it goes.",
  ],
  relatedComponents: ["breadcrumbs"],
  template: {
    baseMarkup: `<a href="{{href}}" class="govuk-back-link">Back</a>`,
    slots: [
      { name: "href", required: true, type: "text", description: "URL to navigate back to" },
    ],
    variants: [],
  },
  accessibilityRules: [
    {
      id: "back-link-placement",
      severity: "warning",
      description: "Back link should appear before the <main> element",
      check: (html) => {
        if (!/govuk-back-link/.test(html)) return { passed: true, message: "No back link present." };
        return { passed: true, message: "Back link present (placement check requires full page context)." };
      },
    },
  ],
};
