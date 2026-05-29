import { ComponentDefinition } from "../types.js";

export const passwordInput: ComponentDefinition = {
  slug: "password-input",
  name: "Password input",
  category: "form",
  description:
    "A text input with a show/hide toggle for entering passwords, with built-in visibility control.",
  useCases: [
    "enter a password",
    "create a password",
    "password field",
    "secure text entry",
    "login credentials",
    "set new password",
  ],
  antiUseCases: [
    "If the field is not for a password or secret, use Text input instead.",
    "If you need to confirm a password, use two Password input components.",
  ],
  relatedComponents: ["text-input"],
  template: {
    baseMarkup: `<div class="govuk-form-group{{errorClass}}">
  <label class="govuk-label" for="{{id}}">
    {{label}}
  </label>
{{hint}}{{error}}  <div class="govuk-password-input" data-module="govuk-password-input">
    <input class="govuk-input govuk-password-input__input govuk-js-password-input-input{{errorInputClass}}" id="{{id}}" name="{{name}}" type="password"{{ariaDescribedBy}} spellcheck="false" autocomplete="{{autocomplete}}" autocapitalize="none">
    <button type="button" class="govuk-button govuk-button--secondary govuk-password-input__toggle govuk-js-password-input-toggle" data-module="govuk-button" aria-controls="{{id}}" aria-label="Show password" hidden>
      Show
    </button>
  </div>
</div>`,
    slots: [
      { name: "label", required: true, type: "text", description: "Visible label text" },
      { name: "hint", required: false, type: "text", description: "Hint text (e.g. password requirements)" },
      { name: "errorMessage", required: false, type: "text", description: "Error message" },
      { name: "id", required: false, type: "text", description: "Element id", default: "password" },
      { name: "name", required: false, type: "text", description: "Form name" },
      { name: "autocomplete", required: false, type: "text", description: "Autocomplete value: current-password or new-password", default: "current-password" },
    ],
    variants: [
      { name: "new-password", description: "For creating passwords (autocomplete=\"new-password\")" },
      { name: "with-error", description: "Password in error state" },
    ],
  },
  accessibilityRules: [
    {
      id: "password-spellcheck-off",
      severity: "warning",
      description: "Password inputs should have spellcheck=\"false\" to prevent password exposure",
      check: (html) => {
        const hasPasswordInput = /type=["']password["']/.test(html);
        if (!hasPasswordInput) return { passed: true, message: "No password input present." };
        if (!/spellcheck=["']false["']/.test(html)) {
          return { passed: false, message: "Password input missing spellcheck=\"false\".", suggestion: "Add spellcheck=\"false\" to prevent spell-checkers from exposing the password." };
        }
        return { passed: true, message: "Password input has spellcheck disabled." };
      },
    },
  ],
};
