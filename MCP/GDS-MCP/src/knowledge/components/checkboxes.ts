import { ComponentDefinition } from "../types.js";

export const checkboxes: ComponentDefinition = {
  slug: "checkboxes",
  name: "Checkboxes",
  category: "form",
  description:
    "Let users select one or more options from a list using checkboxes grouped in a fieldset.",
  useCases: [
    "select multiple options",
    "choose more than one",
    "tick boxes",
    "multi-select from a list",
    "select all that apply",
    "toggle options on and off",
    "pick several items",
  ],
  antiUseCases: [
    "If users can only select one option, use Radios instead.",
    "If the list is very long (more than ~10 items), consider letting users filter or search.",
    "If there's only one checkbox for a standalone yes/no toggle, still use this component.",
  ],
  relatedComponents: ["radios", "select"],
  template: {
    baseMarkup: `<div class="govuk-form-group{{errorClass}}">
  <fieldset class="govuk-fieldset"{{ariaDescribedBy}}>
    <legend class="govuk-fieldset__legend{{legendClass}}">
      {{legend}}
    </legend>
{{hint}}{{error}}    <div class="govuk-checkboxes" data-module="govuk-checkboxes">
{{items}}
    </div>
  </fieldset>
</div>`,
    slots: [
      { name: "legend", required: true, type: "text", description: "The fieldset legend (question text)" },
      { name: "hint", required: false, type: "text", description: "Hint text for the group" },
      { name: "errorMessage", required: false, type: "text", description: "Error message" },
      { name: "items", required: true, type: "array", description: "Checkbox items with text and value" },
      { name: "id", required: false, type: "text", description: "Base id for the group", default: "checkboxes" },
      { name: "name", required: false, type: "text", description: "Form name attribute" },
    ],
    variants: [
      { name: "with-hint", description: "Checkboxes with group-level hint" },
      { name: "with-error", description: "Checkboxes in error state" },
      { name: "small", description: "Smaller checkboxes for filters or secondary UI" },
    ],
  },
  accessibilityRules: [
    {
      id: "checkboxes-fieldset-required",
      severity: "error",
      description: "Checkboxes must be wrapped in a fieldset with a legend",
      check: (html) => {
        const hasFieldset = /<fieldset/.test(html);
        const hasLegend = /<legend/.test(html);
        if (!hasFieldset || !hasLegend) {
          return { passed: false, message: "Checkboxes must be inside a <fieldset> with a <legend>.", suggestion: "Wrap checkboxes in <fieldset class=\"govuk-fieldset\"> with a <legend>." };
        }
        return { passed: true, message: "Checkboxes are in a fieldset with legend." };
      },
    },
    {
      id: "checkboxes-items-have-labels",
      severity: "error",
      description: "Each checkbox must have an associated label",
      check: (html) => {
        const inputs = html.match(/<input[^>]*type=["']checkbox["'][^>]*>/g) || [];
        for (const input of inputs) {
          const idMatch = input.match(/id=["']([^"']+)["']/);
          if (!idMatch) {
            return { passed: false, message: "Checkbox input missing id for label association.", suggestion: "Add id to checkbox and a matching <label for=\"[id]\">." };
          }
          const labelPattern = new RegExp(`for=["']${idMatch[1]}["']`);
          if (!labelPattern.test(html)) {
            return { passed: false, message: `Checkbox "${idMatch[1]}" has no associated label.`, suggestion: `Add <label class="govuk-label govuk-checkboxes__label" for="${idMatch[1]}">` };
          }
        }
        return { passed: true, message: "All checkboxes have associated labels." };
      },
    },
  ],
};
