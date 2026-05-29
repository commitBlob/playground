import { ComponentDefinition } from "../types.js";

export const textInput: ComponentDefinition = {
  slug: "text-input",
  name: "Text input",
  category: "form",
  description:
    "A single-line text field for short free-text answers like names, phone numbers, or postcodes.",
  useCases: [
    "enter their name",
    "type a short answer",
    "single line text field",
    "input for name email phone postcode",
    "free text entry short",
    "text box for short answer",
    "ask for one piece of information",
    "collect a reference number",
    "enter an address line",
    "name field",
    "email address input",
    "short text input",
  ],
  antiUseCases: [
    "If users need to enter longer text that might span multiple lines, use Textarea instead.",
    "If users need to select from a fixed set of options, use Radios or Checkboxes instead.",
    "If users need to enter a date, use Date input instead.",
    "If users need to enter a password, use Password input instead.",
  ],
  relatedComponents: ["textarea", "password-input", "date-input"],
  template: {
    baseMarkup: `<div class="govuk-form-group{{errorClass}}">
  <label class="govuk-label" for="{{id}}">
    {{label}}
  </label>
{{hint}}{{error}}  <input class="govuk-input{{inputErrorClass}}{{widthClass}}" id="{{id}}" name="{{name}}" type="{{inputType}}"{{ariaDescribedBy}}>
</div>`,
    slots: [
      {
        name: "label",
        required: true,
        type: "text",
        description: "The visible label text for the input",
      },
      {
        name: "hint",
        required: false,
        type: "text",
        description: "Optional hint text displayed below the label",
      },
      {
        name: "errorMessage",
        required: false,
        type: "text",
        description: "Error message to display when validation fails",
      },
      {
        name: "id",
        required: false,
        type: "text",
        description: "The id attribute for the input element",
        default: "input-1",
      },
      {
        name: "name",
        required: false,
        type: "text",
        description: "The name attribute for form submission",
      },
      {
        name: "inputType",
        required: false,
        type: "text",
        description: "The HTML input type (text, email, tel, etc.)",
        default: "text",
      },
      {
        name: "width",
        required: false,
        type: "text",
        description:
          "Fixed width class: 20, 10, 5, 4, 3, 2 (characters) or full, three-quarters, two-thirds, one-half, one-third, one-quarter",
      },
    ],
    variants: [
      {
        name: "with-hint",
        description: "Text input with hint text below the label",
      },
      {
        name: "with-error",
        description: "Text input in an error state with error message",
      },
      {
        name: "fixed-width",
        description: "Text input with a fixed character width (e.g. width-10 for phone numbers)",
      },
    ],
  },
  accessibilityRules: [
    {
      id: "text-input-label-required",
      severity: "error",
      description: "Text input must have a visible label with matching for/id attributes",
      check: (html: string): { passed: boolean; message: string; suggestion?: string } => {
        const hasLabel = /label[^>]*for=["']([^"']+)["']/.test(html);
        const hasInput = /input[^>]*id=["']([^"']+)["']/.test(html);
        if (!hasLabel || !hasInput) {
          return {
            passed: false,
            message: "Text input is missing a label with a matching 'for' attribute.",
            suggestion:
              'Add a <label class="govuk-label" for="[input-id]"> element before the input.',
          };
        }
        const labelFor = html.match(/label[^>]*for=["']([^"']+)["']/)?.[1];
        const inputId = html.match(/input[^>]*id=["']([^"']+)["']/)?.[1];
        if (labelFor !== inputId) {
          return {
            passed: false,
            message: `Label 'for' attribute ("${labelFor}") does not match input id ("${inputId}").`,
            suggestion: "Ensure the label's 'for' attribute matches the input's 'id' attribute.",
          };
        }
        return { passed: true, message: "Label correctly associated with input." };
      },
    },
    {
      id: "text-input-no-placeholder-only",
      severity: "error",
      description: "Text input must not rely on placeholder as the only label",
      check: (html: string): { passed: boolean; message: string; suggestion?: string } => {
        const hasPlaceholder = /placeholder=["'][^"']+["']/.test(html);
        const hasLabel = /<label[^>]*>/.test(html);
        if (hasPlaceholder && !hasLabel) {
          return {
            passed: false,
            message: "Input uses placeholder text without a visible label.",
            suggestion:
              "Always provide a visible <label> element. Placeholder text disappears when users type and is not accessible.",
          };
        }
        return { passed: true, message: "Input has a visible label (not placeholder-only)." };
      },
    },
    {
      id: "text-input-error-format",
      severity: "warning",
      description:
        "Error messages should include a visually hidden 'Error:' prefix for screen readers",
      check: (html: string): { passed: boolean; message: string; suggestion?: string } => {
        const hasErrorMessage = /govuk-error-message/.test(html);
        if (!hasErrorMessage) {
          return { passed: true, message: "No error message present (not applicable)." };
        }
        const hasVisuallyHidden = /govuk-visually-hidden[^>]*>Error:/.test(html);
        if (!hasVisuallyHidden) {
          return {
            passed: false,
            message: "Error message is missing the visually hidden 'Error:' prefix.",
            suggestion:
              'Add <span class="govuk-visually-hidden">Error:</span> at the start of the error message text.',
          };
        }
        return { passed: true, message: "Error message has visually hidden prefix." };
      },
    },
  ],
};
