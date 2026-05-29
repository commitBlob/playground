import { ComponentDefinition } from "../types.js";

export const fieldset: ComponentDefinition = {
  slug: "fieldset",
  name: "Fieldset",
  category: "content",
  description:
    "Group related form inputs with a legend that describes the group, essential for accessibility.",
  useCases: [
    "group related form fields",
    "fieldset with legend",
    "wrap inputs that belong together",
    "accessible form grouping",
  ],
  antiUseCases: [
    "Don't wrap a single input in a fieldset — just use a label.",
    "Fieldset is already built into Checkboxes, Radios, and Date input — don't double-nest.",
  ],
  relatedComponents: ["checkboxes", "radios", "date-input"],
  template: {
    baseMarkup: `<fieldset class="govuk-fieldset"{{ariaDescribedBy}}>
  <legend class="govuk-fieldset__legend{{legendClass}}">
    {{legend}}
  </legend>
  {{content}}
</fieldset>`,
    slots: [
      { name: "legend", required: true, type: "text", description: "The legend text describing the group" },
      { name: "content", required: true, type: "html", description: "Form fields inside the fieldset" },
      { name: "legendSize", required: false, type: "text", description: "Legend size: s, m, l, xl" },
    ],
    variants: [
      { name: "as-page-heading", description: "Legend styled as page heading (h1 inside legend)" },
    ],
  },
  accessibilityRules: [
    {
      id: "fieldset-must-have-legend",
      severity: "error",
      description: "Fieldsets must contain a legend element",
      check: (html) => {
        if (!/<fieldset/.test(html)) return { passed: true, message: "No fieldset present." };
        if (!/<legend/.test(html)) {
          return { passed: false, message: "Fieldset is missing a <legend> element.", suggestion: "Add <legend class=\"govuk-fieldset__legend\">Group description</legend> inside the fieldset." };
        }
        return { passed: true, message: "Fieldset has a legend." };
      },
    },
  ],
};
