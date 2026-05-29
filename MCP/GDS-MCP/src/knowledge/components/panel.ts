import { ComponentDefinition } from "../types.js";

export const panel: ComponentDefinition = {
  slug: "panel",
  name: "Panel",
  category: "messaging",
  description:
    "A large green confirmation panel used on transaction complete pages to confirm a successful submission.",
  useCases: [
    "confirmation page",
    "application submitted successfully",
    "transaction complete",
    "success panel with reference number",
    "application complete confirmation",
    "submitted confirmation page",
    "final success page",
  ],
  antiUseCases: [
    "Don't use mid-flow — only on the final confirmation page.",
    "For non-final success messages, use Notification banner with success type.",
  ],
  relatedComponents: ["notification-banner"],
  template: {
    baseMarkup: `<div class="govuk-panel govuk-panel--confirmation">
  <h1 class="govuk-panel__title">
    {{title}}
  </h1>
  <div class="govuk-panel__body">
    {{body}}
  </div>
</div>`,
    slots: [
      { name: "title", required: true, type: "text", description: "Main confirmation heading (e.g. 'Application complete')" },
      { name: "body", required: false, type: "html", description: "Panel body (e.g. reference number)" },
    ],
    variants: [],
  },
  accessibilityRules: [],
};
