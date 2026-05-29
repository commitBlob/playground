import { ComponentDefinition } from "../types.js";

export const table: ComponentDefinition = {
  slug: "table",
  name: "Table",
  category: "content",
  description:
    "Display tabular data in rows and columns with proper header cells and scope attributes.",
  useCases: [
    "display data in rows and columns",
    "tabular data",
    "comparison table",
    "data grid",
    "list of records with multiple fields",
  ],
  antiUseCases: [
    "Don't use tables for layout purposes.",
    "For key-value pairs (not tabular data), use Summary list instead.",
    "If the table would have only one column, use a list instead.",
  ],
  relatedComponents: ["summary-list"],
  template: {
    baseMarkup: `<table class="govuk-table">
  <caption class="govuk-table__caption govuk-table__caption--m">{{caption}}</caption>
  <thead class="govuk-table__head">
    <tr class="govuk-table__row">
{{headCells}}
    </tr>
  </thead>
  <tbody class="govuk-table__body">
{{bodyRows}}
  </tbody>
</table>`,
    slots: [
      { name: "caption", required: true, type: "text", description: "Table caption describing the data" },
      { name: "head", required: true, type: "array", description: "Column headers" },
      { name: "rows", required: true, type: "array", description: "Data rows" },
    ],
    variants: [],
  },
  accessibilityRules: [
    {
      id: "table-caption",
      severity: "warning",
      description: "Tables should have a caption element",
      check: (html) => {
        if (!/govuk-table/.test(html) && !/<table/.test(html)) return { passed: true, message: "No table present." };
        if (!/<caption/.test(html)) {
          return { passed: false, message: "Table is missing a <caption> element.", suggestion: "Add <caption class=\"govuk-table__caption\">Description</caption> after <table>." };
        }
        return { passed: true, message: "Table has a caption." };
      },
    },
    {
      id: "table-header-scope",
      severity: "warning",
      description: "Table header cells should have scope attributes",
      check: (html) => {
        if (!/<th/.test(html)) return { passed: true, message: "No table headers present." };
        const ths = html.match(/<th[^>]*>/g) || [];
        for (const th of ths) {
          if (!/scope=["']/.test(th)) {
            return { passed: false, message: "Table <th> missing scope attribute.", suggestion: "Add scope=\"col\" for column headers or scope=\"row\" for row headers." };
          }
        }
        return { passed: true, message: "All table headers have scope." };
      },
    },
  ],
};
