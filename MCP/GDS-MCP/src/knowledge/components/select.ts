import { ComponentDefinition } from "../types.js";

export const select: ComponentDefinition = {
  slug: "select",
  name: "Select",
  category: "form",
  description:
    "A dropdown list for selecting one option from many. Use sparingly — radios are generally better for fewer options.",
  useCases: [
    "dropdown list",
    "select from many options",
    "long list of choices",
    "country picker",
    "sort order selection",
    "filter by category",
  ],
  antiUseCases: [
    "If there are fewer than 8 options, use Radios instead — they're more accessible and easier to use.",
    "If users can select multiple items, use Checkboxes instead.",
    "Research shows selects are hard for some users — only use when you've tested with your users.",
  ],
  relatedComponents: ["radios", "checkboxes"],
  template: {
    baseMarkup: `<div class="govuk-form-group{{errorClass}}">
  <label class="govuk-label" for="{{id}}">
    {{label}}
  </label>
{{hint}}{{error}}  <select class="govuk-select{{errorInputClass}}" id="{{id}}" name="{{name}}"{{ariaDescribedBy}}>
{{options}}
  </select>
</div>`,
    slots: [
      { name: "label", required: true, type: "text", description: "Visible label text" },
      { name: "hint", required: false, type: "text", description: "Hint text" },
      { name: "errorMessage", required: false, type: "text", description: "Error message" },
      { name: "items", required: true, type: "array", description: "Options with text and value" },
      { name: "id", required: false, type: "text", description: "Element id", default: "select-1" },
      { name: "name", required: false, type: "text", description: "Form name" },
    ],
    variants: [
      { name: "with-error", description: "Select in error state" },
    ],
  },
  accessibilityRules: [
    {
      id: "select-label-required",
      severity: "error",
      description: "Select must have an associated label",
      check: (html) => {
        const hasSelect = /<select[^>]*id=["']([^"']+)["']/.test(html);
        const hasLabel = /label[^>]*for=["']([^"']+)["']/.test(html);
        if (hasSelect && !hasLabel) {
          return { passed: false, message: "Select element missing associated label.", suggestion: "Add a <label for=\"[id]\"> element." };
        }
        return { passed: true, message: "Select has associated label." };
      },
    },
  ],
};
