import { ComponentDefinition } from "../types.js";

export const characterCount: ComponentDefinition = {
  slug: "character-count",
  name: "Character count",
  category: "form",
  description:
    "A textarea with a character or word count that updates as users type, helping them stay within limits.",
  useCases: [
    "limit text length",
    "character limit on textarea",
    "word count",
    "maximum characters allowed",
    "text with length restriction",
    "textarea with counter",
  ],
  antiUseCases: [
    "If there's no meaningful character or word limit, use a plain Textarea instead.",
    "If the limit is very short (under 20 characters), use Text input with a width constraint instead.",
  ],
  relatedComponents: ["textarea", "text-input"],
  template: {
    baseMarkup: `<div class="govuk-character-count" data-module="govuk-character-count" data-maxlength="{{maxlength}}">
  <div class="govuk-form-group{{errorClass}}">
    <label class="govuk-label" for="{{id}}">
      {{label}}
    </label>
{{hint}}{{error}}    <textarea class="govuk-textarea govuk-js-character-count{{errorInputClass}}" id="{{id}}" name="{{name}}" rows="{{rows}}"{{ariaDescribedBy}} aria-describedby="{{id}}-info"></textarea>
  </div>
  <div id="{{id}}-info" class="govuk-hint govuk-character-count__message">
    You can enter up to {{maxlength}} characters
  </div>
</div>`,
    slots: [
      { name: "label", required: true, type: "text", description: "Visible label text" },
      { name: "maxlength", required: true, type: "text", description: "Maximum number of characters" },
      { name: "hint", required: false, type: "text", description: "Hint text" },
      { name: "errorMessage", required: false, type: "text", description: "Error message" },
      { name: "id", required: false, type: "text", description: "Element id", default: "with-hint" },
      { name: "name", required: false, type: "text", description: "Form name" },
      { name: "rows", required: false, type: "text", description: "Visible rows", default: "5" },
    ],
    variants: [
      { name: "with-word-count", description: "Count words instead of characters (use data-maxwords)" },
      { name: "with-threshold", description: "Only show count after threshold percentage reached" },
    ],
  },
  accessibilityRules: [
    {
      id: "character-count-info-region",
      severity: "warning",
      description: "Character count message should be linked via aria-describedby",
      check: (html) => {
        const hasInfo = /character-count__message/.test(html);
        if (!hasInfo) return { passed: true, message: "No character count message present." };
        const hasAriaLink = /aria-describedby[^"]*-info/.test(html);
        if (!hasAriaLink) {
          return { passed: false, message: "Character count message not linked to textarea.", suggestion: "Add the info element's id to the textarea's aria-describedby." };
        }
        return { passed: true, message: "Character count info linked via aria-describedby." };
      },
    },
  ],
};
