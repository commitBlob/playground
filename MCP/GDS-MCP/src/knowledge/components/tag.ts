import { ComponentDefinition } from "../types.js";

export const tag: ComponentDefinition = {
  slug: "tag",
  name: "Tag",
  category: "content",
  description:
    "A coloured label used to indicate status, like 'Completed', 'In progress', or 'Not started'.",
  useCases: [
    "status indicator",
    "label for state",
    "completed in progress tag",
    "coloured status badge",
    "phase indicator",
  ],
  antiUseCases: [
    "Don't use colour alone to convey meaning — the text must communicate the status.",
    "For long descriptions, use a sentence instead of a tag.",
  ],
  relatedComponents: ["task-list", "phase-banner"],
  template: {
    baseMarkup: `<strong class="govuk-tag{{colourClass}}">
  {{text}}
</strong>`,
    slots: [
      { name: "text", required: true, type: "text", description: "The tag text" },
      { name: "colour", required: false, type: "text", description: "Colour variant: grey, green, turquoise, blue, light-blue, purple, pink, red, orange, yellow" },
    ],
    variants: [],
  },
  accessibilityRules: [],
};
