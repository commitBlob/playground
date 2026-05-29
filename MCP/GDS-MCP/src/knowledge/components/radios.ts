import { ComponentDefinition } from "../types.js";

export const radios: ComponentDefinition = {
  slug: "radios",
  name: "Radios",
  category: "form",
  description:
    "Let users select a single option from a list using radio buttons grouped in a fieldset.",
  useCases: [
    "select one option from a list",
    "choose one thing",
    "single selection",
    "pick one from several options",
    "yes or no question",
    "either or choice",
    "mutually exclusive options",
  ],
  antiUseCases: [
    "If users can select more than one option, use Checkboxes instead.",
    "If there are more than ~8 options, consider using Select (but prefer radios for 2-5 options).",
    "Do not pre-select a radio option — users might miss the question and submit a wrong answer.",
  ],
  relatedComponents: ["checkboxes", "select"],
  template: {
    baseMarkup: `<div class="govuk-form-group{{errorClass}}">
  <fieldset class="govuk-fieldset"{{ariaDescribedBy}}>
    <legend class="govuk-fieldset__legend{{legendClass}}">
      {{legend}}
    </legend>
{{hint}}{{error}}    <div class="govuk-radios" data-module="govuk-radios">
{{items}}
    </div>
  </fieldset>
</div>`,
    slots: [
      { name: "legend", required: true, type: "text", description: "The fieldset legend (question text)" },
      { name: "hint", required: false, type: "text", description: "Hint text for the group" },
      { name: "errorMessage", required: false, type: "text", description: "Error message" },
      { name: "items", required: true, type: "array", description: "Radio items with text and value" },
      { name: "id", required: false, type: "text", description: "Base id for the group", default: "radio" },
      { name: "name", required: false, type: "text", description: "Form name attribute" },
    ],
    variants: [
      { name: "with-hint", description: "Radios with group-level hint" },
      { name: "with-error", description: "Radios in error state" },
      { name: "inline", description: "Radios displayed inline (for 2 options like Yes/No)" },
      { name: "small", description: "Smaller radios for filters" },
    ],
  },
  accessibilityRules: [
    {
      id: "radios-fieldset-required",
      severity: "error",
      description: "Radios must be wrapped in a fieldset with a legend",
      check: (html) => {
        const hasFieldset = /<fieldset/.test(html);
        const hasLegend = /<legend/.test(html);
        if (!hasFieldset || !hasLegend) {
          return { passed: false, message: "Radios must be inside a <fieldset> with a <legend>.", suggestion: "Wrap radios in <fieldset class=\"govuk-fieldset\"> with a <legend>." };
        }
        return { passed: true, message: "Radios are in a fieldset with legend." };
      },
    },
    {
      id: "radios-no-preselect",
      severity: "warning",
      description: "Radio buttons should not be pre-selected — users may miss the question",
      check: (html) => {
        const hasChecked = /<input[^>]*type=["']radio["'][^>]*checked/.test(html);
        if (hasChecked) {
          return { passed: false, message: "A radio is pre-selected. Users may submit without reviewing.", suggestion: "Remove the 'checked' attribute so users must actively choose." };
        }
        return { passed: true, message: "No radios are pre-selected." };
      },
    },
  ],
};
