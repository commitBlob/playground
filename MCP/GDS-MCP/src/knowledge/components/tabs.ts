import { ComponentDefinition } from "../types.js";

export const tabs: ComponentDefinition = {
  slug: "tabs",
  name: "Tabs",
  category: "navigation",
  description:
    "Switch between related sections of content within a single page, showing one panel at a time.",
  useCases: [
    "switch between content sections",
    "tabbed interface",
    "related content panels",
    "different views of the same data",
    "organise content into tabs",
  ],
  antiUseCases: [
    "Don't use tabs for sequential content that users should read in order.",
    "On mobile, tabs stack vertically — if content is long, consider separate pages.",
    "If users need to compare content between tabs, show it on the same page without tabs.",
  ],
  relatedComponents: ["accordion"],
  template: {
    baseMarkup: `<div class="govuk-tabs" data-module="govuk-tabs">
  <h2 class="govuk-tabs__title">
    {{title}}
  </h2>
  <ul class="govuk-tabs__list">
{{tabItems}}
  </ul>
{{tabPanels}}
</div>`,
    slots: [
      { name: "title", required: false, type: "text", description: "Title shown above tabs (for screen readers)", default: "Contents" },
      { name: "items", required: true, type: "array", description: "Tab items with label and panel content" },
    ],
    variants: [],
  },
  accessibilityRules: [
    {
      id: "tabs-title-present",
      severity: "warning",
      description: "Tabs should have a title element for screen readers",
      check: (html) => {
        if (!/govuk-tabs/.test(html)) return { passed: true, message: "No tabs present." };
        if (!/govuk-tabs__title/.test(html)) {
          return { passed: false, message: "Tabs missing title heading.", suggestion: "Add <h2 class=\"govuk-tabs__title\">Contents</h2> before the tab list." };
        }
        return { passed: true, message: "Tabs have a title element." };
      },
    },
  ],
};
