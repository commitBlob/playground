import { ComponentDefinition } from "../types.js";

export const errorSummary: ComponentDefinition = {
  slug: "error-summary",
  name: "Error summary",
  category: "messaging",
  description:
    "A summary box at the top of the page listing all form errors with links to the affected fields.",
  useCases: [
    "list of all errors on page",
    "error summary at top",
    "form validation summary",
    "link to error fields",
  ],
  antiUseCases: [
    "Don't use without inline Error messages next to each field.",
    "Don't use for success — use Panel or Notification banner.",
  ],
  relatedComponents: ["error-message"],
  template: {
    baseMarkup: `<div class="govuk-error-summary" data-module="govuk-error-summary">
  <div role="alert">
    <h2 class="govuk-error-summary__title">
      There is a problem
    </h2>
    <div class="govuk-error-summary__body">
      <ul class="govuk-list govuk-error-summary__list">
{{errorLinks}}
      </ul>
    </div>
  </div>
</div>`,
    slots: [
      { name: "errors", required: true, type: "array", description: "Error items with text and href linking to the field" },
    ],
    variants: [],
  },
  accessibilityRules: [
    {
      id: "error-summary-role-alert",
      severity: "error",
      description: "Error summary must include role=\"alert\" for immediate screen reader announcement",
      check: (html) => {
        if (!/govuk-error-summary/.test(html)) return { passed: true, message: "No error summary present." };
        if (!/role=["']alert["']/.test(html)) {
          return { passed: false, message: "Error summary missing role=\"alert\".", suggestion: "Add role=\"alert\" to the container div inside the error summary." };
        }
        return { passed: true, message: "Error summary has role=\"alert\"." };
      },
    },
    {
      id: "error-summary-links-to-fields",
      severity: "warning",
      description: "Error summary items should link to the affected form fields",
      check: (html) => {
        if (!/govuk-error-summary/.test(html)) return { passed: true, message: "No error summary present." };
        if (!/<a[^>]*href=["']#/.test(html)) {
          return { passed: false, message: "Error summary has no links to form fields.", suggestion: "Each error should link to the field: <a href=\"#field-id\">Error text</a>." };
        }
        return { passed: true, message: "Error summary items link to fields." };
      },
    },
  ],
};
