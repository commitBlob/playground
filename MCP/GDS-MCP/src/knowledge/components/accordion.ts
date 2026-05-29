import { ComponentDefinition } from "../types.js";

export const accordion: ComponentDefinition = {
  slug: "accordion",
  name: "Accordion",
  category: "navigation",
  description:
    "Vertically stacked sections that expand and collapse, letting users show and hide related content.",
  useCases: [
    "show and hide sections",
    "collapsible content sections",
    "expandable FAQ",
    "long page with distinct sections",
    "reduce page length by hiding content",
    "grouped information that users can browse",
  ],
  antiUseCases: [
    "If users need to see all content at once, don't hide it in an accordion.",
    "If there's only one section to expand, use the Details component instead.",
    "Don't use accordions to split a sequential form — use separate pages.",
  ],
  relatedComponents: ["details", "tabs"],
  template: {
    baseMarkup: `<div class="govuk-accordion" data-module="govuk-accordion" id="{{id}}">
{{sections}}
</div>`,
    slots: [
      { name: "id", required: true, type: "text", description: "Unique id for the accordion" },
      { name: "sections", required: true, type: "array", description: "Section objects with heading and content" },
    ],
    variants: [],
  },
  accessibilityRules: [
    {
      id: "accordion-unique-id",
      severity: "error",
      description: "Accordion must have a unique id attribute",
      check: (html) => {
        if (!/govuk-accordion/.test(html)) return { passed: true, message: "No accordion present." };
        if (!/<div[^>]*class="[^"]*govuk-accordion[^"]*"[^>]*id=["']/.test(html)) {
          return { passed: false, message: "Accordion container missing id attribute.", suggestion: "Add a unique id to the accordion div." };
        }
        return { passed: true, message: "Accordion has an id attribute." };
      },
    },
  ],
};
