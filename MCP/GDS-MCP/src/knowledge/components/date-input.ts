import { ComponentDefinition } from "../types.js";

export const dateInput: ComponentDefinition = {
  slug: "date-input",
  name: "Date input",
  category: "form",
  description:
    "Three separate text inputs for day, month, and year, grouped in a fieldset for memorable dates.",
  useCases: [
    "enter a date",
    "date of birth",
    "day month year input",
    "when did something happen",
    "passport issue date",
    "memorable date",
    "date fields",
  ],
  antiUseCases: [
    "If users need to pick from available dates (like booking), use a custom date picker instead.",
    "If only an approximate date is needed (month and year), adapt to show only those fields.",
    "Do not use a calendar widget for memorable dates — users know the date and can type it faster.",
  ],
  relatedComponents: ["text-input"],
  template: {
    baseMarkup: `<div class="govuk-form-group{{errorClass}}">
  <fieldset class="govuk-fieldset" role="group"{{ariaDescribedBy}}>
    <legend class="govuk-fieldset__legend{{legendClass}}">
      {{legend}}
    </legend>
{{hint}}{{error}}    <div class="govuk-date-input" id="{{id}}">
      <div class="govuk-date-input__item">
        <div class="govuk-form-group">
          <label class="govuk-label govuk-date-input__label" for="{{id}}-day">
            Day
          </label>
          <input class="govuk-input govuk-date-input__input govuk-input--width-2{{dayErrorClass}}" id="{{id}}-day" name="{{name}}-day" type="text" inputmode="numeric">
        </div>
      </div>
      <div class="govuk-date-input__item">
        <div class="govuk-form-group">
          <label class="govuk-label govuk-date-input__label" for="{{id}}-month">
            Month
          </label>
          <input class="govuk-input govuk-date-input__input govuk-input--width-2{{monthErrorClass}}" id="{{id}}-month" name="{{name}}-month" type="text" inputmode="numeric">
        </div>
      </div>
      <div class="govuk-date-input__item">
        <div class="govuk-form-group">
          <label class="govuk-label govuk-date-input__label" for="{{id}}-year">
            Year
          </label>
          <input class="govuk-input govuk-date-input__input govuk-input--width-4{{yearErrorClass}}" id="{{id}}-year" name="{{name}}-year" type="text" inputmode="numeric">
        </div>
      </div>
    </div>
  </fieldset>
</div>`,
    slots: [
      { name: "legend", required: true, type: "text", description: "The fieldset legend (e.g. 'What is your date of birth?')" },
      { name: "hint", required: false, type: "text", description: "Hint text (e.g. 'For example, 27 3 2007')" },
      { name: "errorMessage", required: false, type: "text", description: "Error message" },
      { name: "id", required: false, type: "text", description: "Base id prefix", default: "date" },
      { name: "name", required: false, type: "text", description: "Base name prefix" },
    ],
    variants: [
      { name: "with-error-on-day", description: "Error highlighting on the day field only" },
      { name: "with-error-on-all", description: "Error highlighting on all fields" },
      { name: "month-year-only", description: "Only month and year fields (no day)" },
    ],
  },
  accessibilityRules: [
    {
      id: "date-input-role-group",
      severity: "warning",
      description: "Date input fieldset should have role=\"group\" for screen readers",
      check: (html) => {
        const hasDateInput = /govuk-date-input/.test(html);
        if (!hasDateInput) return { passed: true, message: "No date input present." };
        const hasRoleGroup = /role=["']group["']/.test(html);
        if (!hasRoleGroup) {
          return { passed: false, message: "Date input fieldset missing role=\"group\".", suggestion: "Add role=\"group\" to the fieldset containing date inputs." };
        }
        return { passed: true, message: "Date input has role=\"group\"." };
      },
    },
    {
      id: "date-input-inputmode",
      severity: "warning",
      description: "Date input fields should use inputmode=\"numeric\" for mobile keyboards",
      check: (html) => {
        const hasDateInput = /govuk-date-input/.test(html);
        if (!hasDateInput) return { passed: true, message: "No date input present." };
        const inputs = html.match(/<input[^>]*govuk-date-input__input[^>]*>/g) || [];
        for (const input of inputs) {
          if (!/inputmode=["']numeric["']/.test(input)) {
            return { passed: false, message: "Date input field missing inputmode=\"numeric\".", suggestion: "Add inputmode=\"numeric\" to date input fields for mobile numeric keyboard." };
          }
        }
        return { passed: true, message: "Date input fields have inputmode=\"numeric\"." };
      },
    },
  ],
};
