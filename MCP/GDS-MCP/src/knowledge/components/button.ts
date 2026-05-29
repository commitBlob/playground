import { ComponentDefinition } from "../types.js";

export const button: ComponentDefinition = {
  slug: "button",
  name: "Button",
  category: "action",
  description:
    "A clickable button for submitting forms or triggering actions, with variants for primary, secondary, warning, and start.",
  useCases: [
    "submit a form",
    "call to action",
    "trigger an action",
    "start button to begin service",
    "save and continue",
    "primary action button",
    "secondary action",
    "destructive action",
  ],
  antiUseCases: [
    "If the action navigates to a new page without side effects, use a link instead.",
    "Don't disable buttons without explaining why — users can't tell what's wrong.",
    "Don't use multiple primary buttons on the same page.",
  ],
  relatedComponents: [],
  template: {
    baseMarkup: `<button type="{{type}}" class="govuk-button{{variantClass}}" data-module="govuk-button"{{disabled}}>
  {{text}}
</button>`,
    slots: [
      { name: "text", required: true, type: "text", description: "Button text (e.g. 'Save and continue')" },
      { name: "type", required: false, type: "text", description: "Button type: submit, button", default: "submit" },
      { name: "variant", required: false, type: "text", description: "Variant: secondary, warning, start, inverse" },
      { name: "disabled", required: false, type: "boolean", description: "Whether the button is disabled" },
    ],
    variants: [
      { name: "secondary", description: "Grey secondary button for less important actions" },
      { name: "warning", description: "Red warning button for destructive actions" },
      { name: "start", description: "Green start button with arrow for beginning a service" },
      { name: "inverse", description: "White button for use on dark backgrounds" },
    ],
  },
  accessibilityRules: [
    {
      id: "button-data-module",
      severity: "warning",
      description: "GOV.UK buttons should have data-module=\"govuk-button\" for double-click prevention",
      check: (html) => {
        if (!/govuk-button/.test(html)) return { passed: true, message: "No GOV.UK button present." };
        if (!/data-module=["']govuk-button["']/.test(html)) {
          return { passed: false, message: "Button missing data-module=\"govuk-button\".", suggestion: "Add data-module=\"govuk-button\" for double-click prevention JavaScript." };
        }
        return { passed: true, message: "Button has data-module." };
      },
    },
    {
      id: "button-disabled-aria",
      severity: "warning",
      description: "Disabled buttons should use aria-disabled alongside disabled attribute",
      check: (html) => {
        if (!/govuk-button/.test(html)) return { passed: true, message: "No button present." };
        const hasDisabled = /\bdisabled\b/.test(html);
        if (hasDisabled && !/aria-disabled=["']true["']/.test(html)) {
          return { passed: false, message: "Disabled button missing aria-disabled=\"true\".", suggestion: "Add aria-disabled=\"true\" alongside the disabled attribute." };
        }
        return { passed: true, message: "Button disability state correctly communicated." };
      },
    },
  ],
};
