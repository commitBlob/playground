import { ComponentDefinition } from "../types.js";

export const textarea: ComponentDefinition = {
  slug: "textarea",
  name: "Textarea",
  category: "form",
  description:
    "A multi-line text field for longer free-text answers like feedback, descriptions, or additional details.",
  useCases: [
    "enter longer text",
    "multi-line text input",
    "free text feedback",
    "describe something in detail",
    "provide additional information",
    "comments or notes",
    "enter a message",
    "provide more detail",
  ],
  antiUseCases: [
    "If users only need a short single-line answer, use Text input instead.",
    "If you need to limit character count with visible feedback, use Character count instead.",
  ],
  relatedComponents: ["text-input", "character-count"],
  template: {
    baseMarkup: `<div class="govuk-form-group{{errorClass}}">
  <label class="govuk-label" for="{{id}}">
    {{label}}
  </label>
{{hint}}{{error}}  <textarea class="govuk-textarea{{errorInputClass}}" id="{{id}}" name="{{name}}" rows="{{rows}}"{{ariaDescribedBy}}></textarea>
</div>`,
    slots: [
      { name: "label", required: true, type: "text", description: "Visible label text" },
      { name: "hint", required: false, type: "text", description: "Hint text below the label" },
      { name: "errorMessage", required: false, type: "text", description: "Error message text" },
      { name: "id", required: false, type: "text", description: "Element id", default: "more-detail" },
      { name: "name", required: false, type: "text", description: "Form submission name" },
      { name: "rows", required: false, type: "text", description: "Number of visible rows", default: "5" },
    ],
    variants: [
      { name: "with-hint", description: "Textarea with hint text" },
      { name: "with-error", description: "Textarea in error state" },
    ],
  },
  accessibilityRules: [
    {
      id: "textarea-label-required",
      severity: "error",
      description: "Textarea must have a visible label with matching for/id",
      check: (html) => {
        const hasLabel = /label[^>]*for=["']([^"']+)["']/.test(html);
        const hasTextarea = /textarea[^>]*id=["']([^"']+)["']/.test(html);
        if (!hasLabel || !hasTextarea) {
          return { passed: false, message: "Textarea missing associated label.", suggestion: "Add a <label for=\"[id]\"> element." };
        }
        return { passed: true, message: "Textarea has associated label." };
      },
    },
  ],
};
