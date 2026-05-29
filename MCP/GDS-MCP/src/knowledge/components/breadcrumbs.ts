import { ComponentDefinition } from "../types.js";

export const breadcrumbs: ComponentDefinition = {
  slug: "breadcrumbs",
  name: "Breadcrumbs",
  category: "navigation",
  description:
    "A navigation trail showing where the current page sits in the site hierarchy.",
  useCases: [
    "show page hierarchy",
    "breadcrumb navigation",
    "site structure trail",
    "parent page links",
    "navigate up the site tree",
  ],
  antiUseCases: [
    "Don't use in a transactional service with linear steps — use Back link instead.",
    "Don't use alongside a back link — choose one.",
    "Don't show breadcrumbs on the homepage.",
  ],
  relatedComponents: ["back-link"],
  template: {
    baseMarkup: `<nav class="govuk-breadcrumbs" aria-label="Breadcrumb">
  <ol class="govuk-breadcrumbs__list">
{{items}}
  </ol>
</nav>`,
    slots: [
      { name: "items", required: true, type: "array", description: "Breadcrumb items with text and optional href" },
    ],
    variants: [
      { name: "collapse-on-mobile", description: "Show only first and last items on mobile" },
    ],
  },
  accessibilityRules: [
    {
      id: "breadcrumbs-aria-label",
      severity: "error",
      description: "Breadcrumbs nav must have aria-label=\"Breadcrumb\"",
      check: (html) => {
        if (!/govuk-breadcrumbs/.test(html)) return { passed: true, message: "No breadcrumbs present." };
        if (!/aria-label=["']Breadcrumb["']/.test(html)) {
          return { passed: false, message: "Breadcrumbs <nav> missing aria-label.", suggestion: "Add aria-label=\"Breadcrumb\" to the <nav> element." };
        }
        return { passed: true, message: "Breadcrumbs have aria-label." };
      },
    },
    {
      id: "breadcrumbs-ordered-list",
      severity: "warning",
      description: "Breadcrumbs should use an ordered list <ol>",
      check: (html) => {
        if (!/govuk-breadcrumbs/.test(html)) return { passed: true, message: "No breadcrumbs present." };
        if (!/<ol/.test(html)) {
          return { passed: false, message: "Breadcrumbs should use <ol> (ordered list) to convey sequence.", suggestion: "Use <ol class=\"govuk-breadcrumbs__list\"> instead of <ul>." };
        }
        return { passed: true, message: "Breadcrumbs use ordered list." };
      },
    },
  ],
};
