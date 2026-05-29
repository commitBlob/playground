import { ComponentDefinition } from "../types.js";

export const insetText: ComponentDefinition = {
  slug: "inset-text",
  name: "Inset text",
  category: "content",
  description:
    "Content with a left border to distinguish it from surrounding text, for supplementary information.",
  useCases: [
    "highlight supplementary content",
    "callout or aside",
    "distinguish a block of text",
    "indented quote or note",
  ],
  antiUseCases: [
    "Don't overuse — if everything is highlighted, nothing stands out.",
    "For critical warnings, use Warning text instead.",
    "For success or important notifications, use Notification banner or Panel.",
  ],
  relatedComponents: ["warning-text", "details", "notification-banner"],
  template: {
    baseMarkup: `<div class="govuk-inset-text">
  {{content}}
</div>`,
    slots: [
      { name: "content", required: true, type: "html", description: "The inset content" },
    ],
    variants: [],
  },
  accessibilityRules: [],
};
