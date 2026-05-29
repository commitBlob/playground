import { ComponentDefinition } from "../types.js";

export const warningText: ComponentDefinition = {
  slug: "warning-text",
  name: "Warning text",
  category: "messaging",
  description:
    "Display a warning with an exclamation icon for consequences that users need to be aware of.",
  useCases: [
    "warn about consequences",
    "important warning message",
    "penalty or legal consequence",
    "caution notice",
    "alert about something serious",
  ],
  antiUseCases: [
    "Don't use for information that isn't genuinely consequential.",
    "For general supplementary info, use Inset text instead.",
  ],
  relatedComponents: ["inset-text", "notification-banner"],
  template: {
    baseMarkup: `<div class="govuk-warning-text">
  <span class="govuk-warning-text__icon" aria-hidden="true">!</span>
  <strong class="govuk-warning-text__text">
    <span class="govuk-visually-hidden">Warning</span>
    {{text}}
  </strong>
</div>`,
    slots: [
      { name: "text", required: true, type: "text", description: "The warning text" },
    ],
    variants: [],
  },
  accessibilityRules: [
    {
      id: "warning-text-hidden-prefix",
      severity: "warning",
      description: "Warning text should include a visually hidden 'Warning' prefix",
      check: (html) => {
        if (!/govuk-warning-text/.test(html)) return { passed: true, message: "No warning text present." };
        if (!/govuk-visually-hidden[^>]*>\s*Warning/.test(html)) {
          return { passed: false, message: "Warning text missing visually hidden 'Warning' prefix.", suggestion: "Add <span class=\"govuk-visually-hidden\">Warning</span> before the text." };
        }
        return { passed: true, message: "Warning text has hidden prefix." };
      },
    },
    {
      id: "warning-text-icon-hidden",
      severity: "warning",
      description: "Warning icon should be hidden from screen readers with aria-hidden",
      check: (html) => {
        if (!/govuk-warning-text__icon/.test(html)) return { passed: true, message: "No warning icon present." };
        if (!/aria-hidden=["']true["']/.test(html)) {
          return { passed: false, message: "Warning icon missing aria-hidden=\"true\".", suggestion: "Add aria-hidden=\"true\" to the icon span." };
        }
        return { passed: true, message: "Warning icon is hidden from assistive tech." };
      },
    },
  ],
};
