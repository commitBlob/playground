import { ComponentDefinition } from "../types.js";

export const details: ComponentDefinition = {
  slug: "details",
  name: "Details",
  category: "content",
  description:
    "A toggleable section that hides supplementary information behind a summary link, using the HTML details element.",
  useCases: [
    "show and hide extra information",
    "expandable help text",
    "supplementary details",
    "hide non-essential content",
    "collapsible section for one item",
  ],
  antiUseCases: [
    "If there are multiple expandable sections, use Accordion instead.",
    "Don't use to hide critical information users need to complete a task.",
  ],
  relatedComponents: ["accordion", "inset-text"],
  template: {
    baseMarkup: `<details class="govuk-details">
  <summary class="govuk-details__summary">
    <span class="govuk-details__summary-text">
      {{summary}}
    </span>
  </summary>
  <div class="govuk-details__text">
    {{content}}
  </div>
</details>`,
    slots: [
      { name: "summary", required: true, type: "text", description: "The visible link text" },
      { name: "content", required: true, type: "html", description: "The expandable content (can be HTML)" },
    ],
    variants: [],
  },
  accessibilityRules: [],
};
