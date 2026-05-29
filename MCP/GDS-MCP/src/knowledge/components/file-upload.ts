import { ComponentDefinition } from "../types.js";

export const fileUpload: ComponentDefinition = {
  slug: "file-upload",
  name: "File upload",
  category: "form",
  description:
    "Let users select and upload a file from their device.",
  useCases: [
    "upload a file",
    "attach a document",
    "submit a photo",
    "file input",
    "upload evidence",
    "attach supporting documents",
  ],
  antiUseCases: [
    "If users need to enter text, use Text input or Textarea instead.",
    "If the upload needs progress feedback or drag-and-drop, you'll need custom JavaScript beyond this component.",
  ],
  relatedComponents: ["text-input"],
  template: {
    baseMarkup: `<div class="govuk-form-group{{errorClass}}">
  <label class="govuk-label" for="{{id}}">
    {{label}}
  </label>
{{hint}}{{error}}  <input class="govuk-file-upload{{errorInputClass}}" id="{{id}}" name="{{name}}" type="file"{{ariaDescribedBy}}>
</div>`,
    slots: [
      { name: "label", required: true, type: "text", description: "Visible label text" },
      { name: "hint", required: false, type: "text", description: "Hint text (e.g. file type/size limits)" },
      { name: "errorMessage", required: false, type: "text", description: "Error message" },
      { name: "id", required: false, type: "text", description: "Element id", default: "file-upload" },
      { name: "name", required: false, type: "text", description: "Form name" },
    ],
    variants: [
      { name: "with-error", description: "File upload in error state" },
    ],
  },
  accessibilityRules: [
    {
      id: "file-upload-label",
      severity: "error",
      description: "File upload must have an associated label",
      check: (html) => {
        const hasLabel = /label[^>]*for=["']([^"']+)["']/.test(html);
        const hasInput = /input[^>]*type=["']file["']/.test(html);
        if (hasInput && !hasLabel) {
          return { passed: false, message: "File upload input missing associated label.", suggestion: "Add a <label for=\"[id]\"> element." };
        }
        return { passed: true, message: "File upload has associated label." };
      },
    },
  ],
};
