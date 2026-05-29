import { ComponentDefinition } from "../types.js";

export const exitThisPage: ComponentDefinition = {
  slug: "exit-this-page",
  name: "Exit this page",
  category: "layout",
  description:
    "A safety button that allows users to quickly leave a page, typically for domestic abuse or sensitive services.",
  useCases: [
    "quick exit button",
    "leave page safely",
    "domestic abuse service exit",
    "escape button for sensitive content",
  ],
  antiUseCases: [
    "Only use for services where safety is a concern (e.g. domestic violence, stalking).",
    "Don't use as a general 'close' or 'cancel' button.",
  ],
  relatedComponents: ["button"],
  template: {
    baseMarkup: `<div class="govuk-exit-this-page" data-module="govuk-exit-this-page">
  <a href="{{href}}" role="button" draggable="false" class="govuk-button govuk-button--warning govuk-exit-this-page__button govuk-js-exit-this-page-button" data-module="govuk-button">
    <span class="govuk-visually-hidden">Emergency</span> Exit this page
  </a>
</div>`,
    slots: [
      { name: "href", required: false, type: "text", description: "URL to redirect to (default: BBC Weather)", default: "https://www.bbc.co.uk/weather" },
    ],
    variants: [],
  },
  accessibilityRules: [
    {
      id: "exit-this-page-role-button",
      severity: "warning",
      description: "Exit this page link should have role=\"button\" since it acts as an action",
      check: (html) => {
        if (!/govuk-exit-this-page/.test(html)) return { passed: true, message: "No exit this page present." };
        if (!/role=["']button["']/.test(html)) {
          return { passed: false, message: "Exit link missing role=\"button\".", suggestion: "Add role=\"button\" to the exit link since it triggers an action." };
        }
        return { passed: true, message: "Exit link has role=\"button\"." };
      },
    },
  ],
};
