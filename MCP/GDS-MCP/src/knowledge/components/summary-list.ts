import { ComponentDefinition } from "../types.js";

export const summaryList: ComponentDefinition = {
  slug: "summary-list",
  name: "Summary list",
  category: "content",
  description:
    "A list of key-value pairs, typically used for 'Check your answers' pages or to summarise information.",
  useCases: [
    "key value pairs",
    "check your answers page",
    "summary of submitted data",
    "display structured information",
    "review before submission",
    "definition list of details",
  ],
  antiUseCases: [
    "For tabular data with multiple rows of the same columns, use Table instead.",
    "For simple lists without keys, use a standard list.",
  ],
  relatedComponents: ["table"],
  template: {
    baseMarkup: `<dl class="govuk-summary-list">
{{rows}}
</dl>`,
    slots: [
      { name: "rows", required: true, type: "array", description: "Rows with key, value, and optional actions" },
    ],
    variants: [
      { name: "no-border", description: "Remove borders between rows" },
      { name: "with-actions", description: "Include Change/Remove action links" },
    ],
  },
  accessibilityRules: [
    {
      id: "summary-list-action-context",
      severity: "warning",
      description: "Action links in summary lists need visually hidden context text",
      check: (html) => {
        if (!/govuk-summary-list/.test(html)) return { passed: true, message: "No summary list present." };
        const links = html.match(/<a[^>]*govuk-link[^>]*>([^<]*)<\/a>/g) || [];
        for (const link of links) {
          if (/>(Change|Remove)<\/a>/.test(link) && !/govuk-visually-hidden/.test(link)) {
            return { passed: false, message: "Action link says 'Change' without visually hidden context.", suggestion: "Add <span class=\"govuk-visually-hidden\"> name</span> after 'Change' so screen readers announce 'Change name'." };
          }
        }
        return { passed: true, message: "Summary list action links have context (or none present)." };
      },
    },
  ],
};
