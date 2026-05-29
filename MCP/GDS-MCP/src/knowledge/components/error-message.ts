import { ComponentDefinition } from "../types.js";

export const errorMessage: ComponentDefinition = {
  slug: "error-message",
  name: "Error message",
  category: "messaging",
  description:
    "An inline error message displayed next to a form field when validation fails.",
  useCases: [
    "field validation error",
    "inline error for input",
    "form field error message",
    "tell user what went wrong",
  ],
  antiUseCases: [
    "Don't use on its own — always pair with Error summary at the top of the page.",
    "Don't use for success messages — use Notification banner or Panel.",
  ],
  relatedComponents: ["error-summary"],
  template: {
    baseMarkup: `<p id="{{id}}-error" class="govuk-error-message">
  <span class="govuk-visually-hidden">Error:</span> {{message}}
</p>`,
    slots: [
      { name: "message", required: true, type: "text", description: "The error message text" },
      { name: "id", required: true, type: "text", description: "Base id matching the input this error relates to" },
    ],
    variants: [],
  },
  accessibilityRules: [
    {
      id: "error-message-hidden-prefix",
      severity: "error",
      description: "Error messages must have a visually hidden 'Error:' prefix",
      check: (html) => {
        if (!/govuk-error-message/.test(html)) return { passed: true, message: "No error message present." };
        if (!/govuk-visually-hidden[^>]*>\s*Error:/.test(html)) {
          return { passed: false, message: "Error message missing visually hidden 'Error:' prefix.", suggestion: "Add <span class=\"govuk-visually-hidden\">Error:</span> before the message text." };
        }
        return { passed: true, message: "Error message has hidden prefix." };
      },
    },
  ],
};
